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

%  ����һ�����ļ��Ի���
[FileName,PathName]=uigetfile('*.txt','ѡ��һ�������ļ���',pwd);
%  ���û��ڶԻ����е����ȡ����ʱ��Open_Callbackֹͣ��
if ( FileName == 0 )
    return
end
%  ��������
[A,C,FileName]=SalesTaxesInput(FileName,PathName);
%  ����
C=ComputeTaxes(C);
%  ��������cell����output��
output=OutputTaxes(A,C);
%  ���������ȫ�ֱ�����
setappdata(0,'output',output);
%  ����������������ȫ�ֱ�����
setappdata(0,'InputFile',FileName);


function [A,C,FileName]=SalesTaxesInput(FileName,PathName)
%  �������ܣ�����Ļ���������ĵ�����(����input1.txt)���������ݡ�
%  ���A������ԭʼ�����ĵ������ݣ�����ã�
%  ���filename������ã�
%  ���C������ͳ�ƺ�����ݣ���������ʹ�ã�

n=0;
i=1;
fid=fopen([PathName,FileName]);
if ( fid == -1 )
    errordlg('�Ҳ��������ļ�','������ʾ');
    return
end
while ( ~feof(fid) )
    A{i,1}=fgets(fid);  % ���ļ���ÿһ�ж���cell����MATLAB��cell���������ƽṹ��
    i=i+1;
    n=n+1;
end
fclose(fid);
C=CountC(A,n);


function C=CountC(A,n)
%  �������ܣ���cell����Aͳ�Ƶõ����C
%  C���кű�ʾ��Ʒ��ţ������i�б�ʾ��i����Ʒ��
%  C���кŷֱ��ʾ����˰�ʣ�����˰�ʣ�����������˰ǰ���ۣ����磺
%  -------------------------------------------
%  | ����˰�� | ����˰�� | �������� | ˰ǰ���� |
%  -------------------------------------------
%  ����'����˰��'Ϊ0��ʾ����Ʒ����book, food, medical product��'����˰��'Ϊ0.1��ʾ������Ʒ��
%  ������'����˰��'Ϊ0�ļ��Ͻ������ˡ�book��,'chocolate','pills'��
%  '����˰��'Ϊ0.1�ļ���������'CD','perfume'��
%  '����˰��'�н�����Ʒ˰��Ϊ0.05���ǽ�����Ʒ˰��Ϊ0��
%  �����߿���GetAttribt��������������'����˰��'�������Լ�ÿ��'����˰��'�а�������Ʒ��

for i=1:n
    str=char(A{i,1});  % ��cellת����char
    str=deblank(str);  %  ȥ����β�Ķ���ո�
    temp=regexp(str,'\s','split');  % �����а��ո񻮷�Ϊ���ɸ�����
    %  ��C��ֵ
    C(i,1)=GetAttribt(str);
    C(i,2)=IsImported(str);
    t1=str2num(char(temp{1,1}));
    t2=str2num(char(temp{1,length(temp)}));
    %  ��������������ʱ����ʾ����ֹ����
    if ( isempty(t1) || isempty(t2) )
        errordlg('�����ļ���ʽ��������������','������ʾ');
        return
    end
    C(i,3)=t1;
    C(i,4)=t2;
end


function y=GetAttribt(str)
%  �������ܣ�CountC�ĸ����������ж���Ʒ������һ�ֻ���˰�ʡ�
%  �������strΪ�ַ������͡�
%  �������book,chocolate,pill�����˰��Ϊ0������Ϊ0.1

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
%  �������ܣ�CountC�ĸ����������ж���Ʒ�Ƿ���ڣ�����������˰�ʡ�
%  �������strΪ�ַ������͡�
%  �������imported��Ϊ0.05������Ϊ0

flag=strfind(str,'imported');
if ( isempty(flag) )  
    y=0;
else
    y=0.05;
end


function C=ComputeTaxes(C)
%  �������ܣ�����table C�����嵥��������Ʒ��˰Ǯ��

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
%  �������ܣ�������ս������cell����Output�С�

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

%  ��ȡ�������
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

%  ��ȡ�������
output=getappdata(0,'output');
InputFile=getappdata(0,'InputFile');
%  ����Ĭ������ļ���
InputFile=['out',InputFile(3:length(InputFile))];
%  ����һ������ļ��Ի���
[FileName,PathName]=uiputfile(InputFile,'����ļ�����Ϊ��');
%  ���û��ڶԻ����е����ȡ����ʱ��Open_Callbackֹͣ��
if ( FileName == 0 )
    return
end
[n,~]=size(output);
%  ���������ļ���
fid=fopen([PathName,FileName],'w');
for i=1:n
    fprintf(fid,'%s\r\n',output{i,1});
end
fclose(fid);
    
