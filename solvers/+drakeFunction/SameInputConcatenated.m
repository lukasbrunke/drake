classdef SameInputConcatenated<drakeFunction.DrakeFunction
  % DrakeFunction representing the concatenation of n functions
  %
  % Implements 
  %
  % \f[
  % f(x) = 
  % \begin{bmatrix}
  %   f_1(x) \\
  %   f_2(x) \\
  %   \vdots \\
  %   f_n(x)
  % \end{bmatrix},\;
  % \frac{df}{dx} = 
  % \begin{bmatrix}
  %   \frac{d f_1}{d x} \\
  %   \frac{d f_2}{d x} \\
  %   \vdots            \\
  %   \frac{d f_n}{d x} 
  % \end{bmatrix}
  % \f]
  properties(SetAccess = immutable)
    contained_functions     % Cell array of DrakeFunction objects
    n_contained_functions   % Number of elements in contained_functions
  end
  
  methods
    function obj = SameInputConcatenated(fcns)
      % obj = Concatenated(fcns, same_input) returns a DrakeFunction
      %   representing the concatenation of a given set of
      %   DrakeFunctions. 
      % @param fcns         -- a cell array of DrakeFunction objects
      typecheck(fcns,'cell');
      typecheck(fcns{1},'drakeFunction.DrakeFunction');
      input_frame = fcns{1}.getInputFrame();
      fcns_length = numel(fcns);
      output_frames = cell(fcns_length,1);
      output_frames{1} = getOutputFrame(fcns{1});
      for i = 2:fcns_length
        typecheck(fcns{i},'drakeFunction.DrakeFunction');
        if(isequal_modulo_transforms(getInputFrame(fcns{i}),input_frame))
          error('Drake:DrakeFunction:InputFramesDoNotMatch','In SameInputConcatenated, the input frames should be the same');
        end
        output_frames{i} = getOutputFrame(fcns{i});
        if(output_frames{i}.dim == 0)
          output_frames{i} = [];
        end
      end
      output_frame = MultiCoordinateFrame.constructFrame(output_frames);
      
      obj = obj@drakeFunction.DrakeFunction(input_frame,output_frame);
      
      obj.contained_functions = fcns;
      obj.n_contained_functions = fcns_length;
      obj = obj.setSparsityPattern();
    end
    
    function [f,df] = eval(obj,varargin)
      % eval(x,{cached_data1,cached_data2,...}) or eval(x)
      % @param x  A column vector. Containing all the x for each drakeFunction
      % @param cached_data_i   A cell, containing all the cached data for the i'th drakeFunction
      x = varargin{1};
      if(nargin>2)
        cached_data_cell = varargin{2};
        [f_cell,df_cell] = evalContainedFunctions(obj,x,cached_data_cell);
      else
        [f_cell,df_cell] = evalContainedFunctions(obj,x);
      end
      [f,df] = combineOutputs(obj,f_cell,df_cell);
    end
  end
end