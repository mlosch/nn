local InputAdaptiveNarrow, parent = torch.class('nn.InputAdaptiveNarrow', 'nn.Module')

function InputAdaptiveNarrow:__init(dimension,borderWidth)
   parent.__init(self)
   self.dimension=dimension
   self.borderWidth=borderWidth
   self.index=0
   self.length=0
end

function InputAdaptiveNarrow:updateOutput(input)
   self.length = input:size(self.dimension) - self.borderWidth
   self.index = math.floor(self.borderWidth/2)
   
   local output=input:narrow(self.dimension,self.index,self.length);
   self.output:resizeAs(output)
   return self.output:copy(output)
end

function InputAdaptiveNarrow:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)  
   self.gradInput:zero();
   self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
   return self.gradInput
end 
