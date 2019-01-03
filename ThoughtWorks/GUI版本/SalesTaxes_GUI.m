function varargout = SalesTaxes_GUI(varargin)
% SALESTAXES_GUI MATLAB code for SalesTaxes_GUI.fig
%      SALESTAXES_GUI, by itself, creates a new SALESTAXES_GUI or raises the existing
%      singleton*.
%
%      H = SALESTAXES_GUI returns the handle to a new SALESTAXES_GUI or the handle to
%      the existing singleton*.
%
%      SALESTAXES_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SALESTAXES_GUI.M with the given input arguments.
%
%      SALESTAXES_GUI('Property','Value',...) creates a new SALESTAXES_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SalesTaxes_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SalesTaxes_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SalesTaxes_GUI

% Last Modified by GUIDE v2.5 24-Oct-2015 21:02:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SalesTaxes_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SalesTaxes_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SalesTaxes_GUI is made visible.
function SalesTaxes_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SalesTaxes_GUI (see VARARGIN)

% Choose default command line output for SalesTaxes_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SalesTaxes_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SalesTaxes_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%  创建一个打开文件对话框。
[FileName,PathName]=uigetfile('*.txt','选择一个输入文件：',pwd);
%  当用户在对话框中点击“取消”时，Open_Callback停止。
if ( FileName == 0 )
    return
end
%  输入数据
[A,C,FileName]=SalesTaxesInput(FileName,PathName);
%  计算
C=ComputeTaxes(C);
%  输出结果到cell矩阵output中
output=OutputTaxes(A,C);
%  将结果存入全局变量中
setappdata(0,'output',output);
%  将输入数据名存入全局变量中
setappdata(0,'InputFile',FileName);


function [A,C,FileName]=SalesTaxesInput(FileName,PathName)
%  函数功能：从屏幕键入输入文档名称(例如input1.txt)并读入数据。
%  输出A，保存原始输入文档的内容，输出用；
%  输出filename，输出用；
%  输出C，保存统计后的数据，后续计算使用；

n=0;
i=1;
fid=fopen([PathName,FileName]);
if ( fid == -1 )
    errordlg('找不到输入文件','错误提示');
    return
end
while ( ~feof(fid) )
    A{i,1}=fgets(fid);  % 将文件的每一行读入cell矩阵，MATLAB中cell的作用类似结构体
    i=i+1;
    n=n+1;
end
fclose(fid);
C=CountC(A,n);


function C=CountC(A,n)
%  函数功能：由cell矩阵A统计得到表格C
%  C的行号表示商品序号，例如第i行表示第i件商品；
%  C的列号分别表示基础税率，进口税率，购买数量，税前单价，例如：
%  -------------------------------------------
%  | 基础税率 | 进口税率 | 购买数量 | 税前单价 |
%  -------------------------------------------
%  其中'基础税率'为0表示：商品属于book, food, medical product；'基础税率'为0.1表示其他商品；
%  本题中'基础税率'为0的集合仅收纳了‘book’,'chocolate','pills'；
%  '基础税率'为0.1的集合收纳了'CD','perfume'；
%  '进口税率'中进口商品税率为0.05，非进口商品税率为0；
%  开发者可在GetAttribt函数中自行扩充'基础税率'的种类以及每种'基础税率'中包含的商品。

for i=1:n
    str=char(A{i,1});  % 从cell转换成char
    str=deblank(str);  %  去掉首尾的多余空格
    temp=regexp(str,'\s','split');  % 将该行按空格划分为若干个部分
    %  给C赋值
    C(i,1)=GetAttribt(str);
    C(i,2)=IsImported(str);
    t1=str2num(char(temp{1,1}));
    t2=str2num(char(temp{1,length(temp)}));
    %  当输入数据有误时，提示并终止程序
    if ( isempty(t1) || isempty(t2) )
        errordlg('输入文件格式有误，请重新输入','错误提示');
        return
    end
    C(i,3)=t1;
    C(i,4)=t2;
end


function y=GetAttribt(str)
%  函数功能：CountC的辅助函数，判断商品属于哪一种基础税率。
%  输入参数str为字符串类型。
%  如果包含book,chocolate,pill则基础税率为0，否则为0.1

compare={'book';'chocolate';'pill'};
for i=1:3
    flag=strfind(str,char(compare{i,1}));
    if ( isempty(flag) )
        flag=0;
    else 
        flag=1;
        break
    end
end
if ( flag == 1 )  
    y=0;
else
    y=0.1;
end


function y=IsImported(str)
%  函数功能：CountC的辅助函数，判断商品是否进口，并给出进口税率。
%  输入参数str为字符串类型。
%  如果包含imported则为0.05，否则为0

flag=strfind(str,'imported');
if ( isempty(flag) )  
    y=0;
else
    y=0.05;
end


function C=ComputeTaxes(C)
%  函数功能：根据table C计算清单上所有商品的税钱。

[n,~]=size(C);
for i=1:n
    t=C(i,1)+C(i,2);
    t=t*C(i,4);
    p=floor(t/0.05);
    if ( p*0.05 ~= t )      
        t=0.05*(p+1);
    end
    C(i,5)=t*C(i,3);
end


function output=OutputTaxes(A,C)
%  函数功能：输出最终结果存入cell矩阵Output中。

[n,~]=size(A);
SalesTaxes=0.0;
Total=0.0;
for i=1:n
    str=char(A{i,1});
    str=deblank(str);
    j=length(str);
    for k=1:2
        while( str(j) ~= ' ' )
            j=j-1;
        end
        j=j-1;
    end
    money=C(i,4)+C(i,5);
    Total=Total+money;
    money=[str(1:j),': ',num2str(money,'%.2f')];
    SalesTaxes=SalesTaxes+C(i,5);
    output{i,1}=money; 
end
output{n+1,1}=['Sales Taxes: ',num2str(SalesTaxes,'%.2f')]; 
output{n+2,1}=['Total: ',num2str(Total,'%.2f')];



% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%  读取输出参数
output=getappdata(0,'output');
[n,~]=size(output);
Str=[];
for i=1:n
    Str=strvcat(Str,char(output{i,1}));
end
set(handles.edit1,'String',Str);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%  读取输出参数
output=getappdata(0,'output');
InputFile=getappdata(0,'InputFile');
%  定义默认输出文件名
InputFile=['out',InputFile(3:length(InputFile))];
%  创建一个输出文件对话框。
[FileName,PathName]=uiputfile(InputFile,'输出文件保存为：');
%  当用户在对话框中点击“取消”时，Open_Callback停止。
if ( FileName == 0 )
    return
end
[n,~]=size(output);
%  输出结果到文件中
fid=fopen([PathName,FileName],'w');
for i=1:n
    fprintf(fid,'%s\r\n',output{i,1});
end
fclose(fid);
    
