require 'xlua'
require 'optim'
require 'cunn'
require 'torch'
require 'image'


--------------------------------------------------------------
-- Data augmentation modules
--------------------------------------------------------------
do -- flip images at random
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')


  local function distortImage(imgInput, width, height, h_sigma, v_sigma)

      h_shift = math.floor(torch.normal(0,h_sigma))
      v_shift = math.floor(torch.normal(0,v_sigma))

      shift=''

      if v_shift > 0 then shift=shift..'b' else shift=shift..'t'  end
      if h_shift > 0 then shift=shift..'r'else shift=shift..'l'  end

      image.crop(imgInput, image.translate(imgInput, imgInput, h_shift, v_shift), shift, (width - h_shift), (height - v_shift))
      image.scale(imgInput, imgInput, width, height)
      return imgInput
  end

  local function rotateImage(imgInput, sigma_theta)
    theta = torch.normal(0, sigma_theta)
    image.rotate(imgInput, imgInput, theta)
    return imgInput
  end

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local h_sigma = 5
      local v_sigma = 5
      local sigma_theta = 0.05

      local imgWidth = input:size(3)
      local imgHeight = input:size(4)

      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
        input[i] = distortImage(input[i], imgWidth, imgHeight, h_sigma, v_sigma)
        input[i] = rotateImage(input[i], sigma_theta)
      end
    end
    self.output:set(input)
    return self.output
  end
end

do  -- distort images 
  local DisortImage, parent = torch.class('nn.DisortImage', 'nn.Module')

  function DisortImage:__init()
    parent.__init(self)
    self.train = true
  end

  function DisortImage:updateOutput(inputs)
    if self.train then

    end
    self.output:set(input)
    return self.output
  end

end

do  -- Add gaussian noise
end

do  -- Rotate images randomly
end

--------------------------------------------------------------
-- Actual train models
--------------------------------------------------------------
