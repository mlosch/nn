local SpatialSampledConvolution, parent = torch.class('nn.SpatialSampledConvolution','nn.Module')

function SpatialSampledConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dkW, dkH, dW, dH, padding)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.hkW = kW
   self.hkH = kH
   self.kW = (dkW+1)*(kW-1)+1
   self.kH = (dkH+1)*(kH-1)+1
   self.dkW = dkW
   self.dkH = dkH

   self.dW = dW or 1
   self.dH = dH or 1

   self.padding = padding or 0

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   self:reset()
end

function SpatialSampledConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.hkW*self.hkH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)  
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
    self._gradOutput = self._gradOutput or gradOutput.new()
    self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
    gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function SpatialSampledConvolution:updateOutput(input)
   input = makeContiguous(self, input)
   return input.nn.SpatialSampledConvolution_updateOutput(self, input)
end

function SpatialSampledConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      return input.nn.SpatialSampledConvolution_updateGradInput(self, input, gradOutput)
   end
end

function SpatialSampledConvolution:accGradParameters(input, gradOutput, scale)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   return input.nn.SpatialSampledConvolution_accGradParameters(self, input, gradOutput, scale)
end

