function [x,Stop]=POP(S,Stop)
if ( StackEmpty(Stop) )
    disp('Stack is empty!');
    return
end
x=S(Stop);
Stop=Stop-1;