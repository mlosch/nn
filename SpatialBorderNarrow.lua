local SpatialBorderNarrow, parent = torch.class('nn.SpatialBorderNarrow', 'nn.Module')

-- Width: 0, Height: 1
function SpatialBorderNarrow:__init(dimension,borderWidth)
   parent.__init(self)
   self.dim = dimension
   self.dimension=nil
   self.borderWidth=borderWidth
   self.index=0
   self.length=0
end

function SpatialBorderNarrow:updateOutput(input)
   self.dimension = input:dim() - self.dim
   self.length = input:size(self.dimension) - self.borderWidth
   self.index = math.floor(self.borderWidth/2)
   
   local output=input:narrow(self.dimension,self.index,self.length);
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function SpatialBorderNarrow:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)  
   self.gradInput:zero();
   self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
   return self.gradInput
end 
