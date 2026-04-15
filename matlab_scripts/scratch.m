classdef mobject
    properties
        val;
        der;
    end
    methods
    function obj = mobject(val_,der_)
        obj.val = val_;
        obj.der = der_;
    end
    function obj_out = plus(obj1_,obj2_)
        obj_out = mobject( ...
        obj1_.val+obj2_.val, ...
        obj1_.der+obj2_.der );
    end
    function obj_out = times(obj1_,obj2_)
        obj_out = mobject( ...
        obj1_.val*obj2_.val, ...
        obj1_.der*obj2_.val+obj1_.val*obj2_.der );
    end



end

function [u,Dx_u] = rosenbrock(x)
    f = @(x_) (1-x_(1))^2+100*(x_(2)-x_(1)^2)^2;
    u = f(x); % u = f(x)
    Dx_u = autodiff(f,x);
end
function [u,Dx_u] = rosenbrock(x)
    [a,b] = deal(1,100); % parameters
    [x1,x2] = deal(x(1),x(2)); % x = [x1 ; x2]
    u = (a-x1)^2 + b*(x2-x1^2)^2; % u = f(x)
    Dx_u = [-2*(a-x1) - 4*b*x1*(x2-x1^2) , 2*b*(x2-x1^2)];
end
function u = rosenbrock(x)
    [a,b] = deal(1,100); % parameters
    [x1,x2] = deal(x(1),x(2)); % x = [x1 ; x2]
    u = (a-x1)^2 + b*(x2-x1^2)^2; % u = f(x)
end

function [u,Dx_u] = f(x)
    u = f_map(x);
    Dx_u = D_f_map(x);
end

function u = f(x)
    u = f_map(x);
end


clear
close all

A_f = eye(size(x,1));
f_map = @(x_) A_f*x_;

function u = f(x)
    u = f_map(x);
end
