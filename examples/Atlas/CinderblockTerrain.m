classdef CinderblockTerrain < RigidBodyTerrain
  %This is just a hack to load the terrain with a cinderblock
  properties(Access = protected)
    cinderblock_size
    cinderblock_xy
  end
  methods
    function obj = CinderblockTerrain(cinderblock_size,cinderblock_xy)
      sizecheck(cinderblock_size,[3,1]);
      sizecheck(cinderblock_xy,[2,1]);
      obj = obj@RigidBodyTerrain();
      obj.cinderblock_size = cinderblock_size;
      obj.cinderblock_xy = cinderblock_xy;
    end
    
    function [z,normal] = getHeight(obj,xy)
      x = xy(1,:);
      y = xy(2,:);
      on_cinderblock = x<= obj.cinderblock_xy(1)+obj.cinderblock_size(1)/2 &...
          x>= obj.cinderblock_xy(1)-obj.cinderblock_size(1)/2 &...
          y<=obj.cinderblock_xy(2)+obj.cinderblock_size(2)/2 &...
          y>=obj.cinderblock_xy(2)-obj.cinderblock_size(2)/2;
      z = zeros(1,size(xy,2));
      z(on_cinderblock) = obj.cinderblock_size(3);
      normal = bsxfun(@times,[0;0;1],ones(1,size(xy,2)));
    end
    
    function geom = getRigidBodyGeometry(obj)
      geom = RigidBodyBox(obj.cinderblock_size,[obj.cinderblock_xy;obj.cinderblock_size(3)/2],[0;0;0]);
    end
    
    function geom = getRigidBodyContactGeometry(obj)
      geom = RigidBodyBox(obj.cinderblock_size,[obj.cinderblock_xy;obj.cinderblock_size(3)/2],[0;0;0]);
    end
    
    function [xgv,ygv] = writeWRL(obj,fp) % for visualization
  
      
      %  color1 = [204 102 0]/256;  % csail orange
      color1 = hex2dec({'ee','cb','ad'})/256;  % something a little brighter (peach puff 2 from http://www.tayloredmktg.com/rgb/)
      color2 = hex2dec({'cd','af','95'})/256;
      fprintf(fp,'Transform {\n  translation %f %f %f\n  children [\n',obj.cinderblock_xy(1),obj.cinderblock_xy(2),obj.cinderblock_size(3)/2);
      fprintf(fp,'Shape { geometry Box {\n');
      %        fprintf(fp,'  solid "false"\n');
      fprintf(fp,'size %f %f %f\n',obj.cinderblock_size(1),obj.cinderblock_size(2),obj.cinderblock_size(3));
      fprintf(fp,'}\n}\n'); % end Shape
      fprintf(fp,']\n}\n'); % end Transform
      fprintf(fp,']\n}\n'); % end Transform
      fprintf(fp,'\n\n');
    end
  end
end