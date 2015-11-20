local SpatialMaxFiltering, parent = torch.class('nn.SpatialMaxFiltering', 'nn.Module')

function SpatialMaxFiltering:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.ceil_mode = false
end

function SpatialMaxFiltering:ceil()
  self.ceil_mode = true
  return self
end

function SpatialMaxFiltering:floor()
  self.ceil_mode = false
  return self
end

function SpatialMaxFiltering:updateOutput(input)
   -- backward compatibility
   self.ceil_mode = self.ceil_mode or false
   input.nn.SpatialMaxFiltering_updateOutput(self, input)
   return self.output
end

function SpatialMaxFiltering:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxFiltering_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialMaxFiltering:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end

function SpatialMaxFiltering:__tostring__()
   local s =  string.format('%s(%d,%d,%d,%d)', torch.type(self),
                            self.kW, self.kH, self.dW, self.dH)
   return s
end
