do -- data augmentation module
  local CentroidFeatures,parent = torch.class('nn.CentroidFeatures', 'nn.SpatialConvolution')

  function CentroidFeatures:__init()
    parent.__init(self)
    self.train = true
  end

  function CentroidFeatures:updateOutput(input)
    self.output:set(input)
    return self.output
  end

  function CentroidFeatures:updateGradInput(input, gradOutput)
    return self.gradInput
  end
end
