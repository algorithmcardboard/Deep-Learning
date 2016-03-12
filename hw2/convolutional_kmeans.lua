require("unsup")

function unsup.kmeans_convoluve(x, k, centroids, kSize, std, niter, batchsize, callback, verbose)

   x = x or error('missing argument: ' .. help)
   k = k or error('missing argument: ' .. help)

   nCentroids = k

   niter = niter or 1
   batchsize = batchsize or math.min(1000, (#x)[1])
   std = std or 0.1

   -- some shortcuts
   local sum = torch.sum
   local max = torch.max
   local pow = torch.pow
   local randn = torch.randn
   local zeros = torch.zeros

   --print("batchSize is " .. batchsize .. " kSize is " .. kSize)

   local numPatches = batchsize/(kSize*kSize)

   -- dims
   local nsamples = (#x)[1]
   local ndims = (#x)[2]

   -- initialize means
   local x2 = sum(pow(x,2),2)
   if not(centroids) then
      centroids = randn(k,ndims)*std
   end
   local totalcounts = zeros(k)

   -- callback?
   if callback then callback(0,centroids:reshape(k_size),totalcounts) end

   -- do niter iterations
   --
   for i = 1,niter do
      -- progress
      if verbose then xlua.progress(i,niter) end

      -- sums of squares
      local c2 = sum(pow(centroids,2),2)*0.5

      -- init some variables
      local summation = zeros(k,ndims)
      local counts = zeros(k)
      local loss = 0

      -- process batch
      for i = 1,nsamples,batchsize do
         -- indices
         local lasti = math.min(i+batchsize-1,nsamples)
         local m = lasti - i + 1

         -- k-means step, on minibatch
         local batch = x[{ {i,lasti},{} }]
         local batch_t = batch:t()
         local tmp = centroids * batch_t
         for n = 1,(#batch)[1] do
            tmp[{ {},n }]:add(-1,c2)
         end

         --print(tmp:size())
         --print(nCentroids .. " " .. numPatches .. " " .. kSize)

         local tmpR = tmp:reshape(nCentroids, numPatches, kSize * kSize)
         local max_vals, max_indices = tmpR:max(3)

         max_vals = max_vals:reshape(nCentroids, numPatches)
         max_indices = max_indices:reshape(nCentroids, numPatches)
         local indices = torch.linspace(0, (numPatches-1)*kSize*kSize, numPatches):float():reshape(1, numPatches):expandAs(max_indices)
         max_indices = indices + max_indices:float()

         local val, labels = max_vals:max(1)

         local ii = torch.eq(max_vals, val:expandAs(max_vals))

         local patchesToSelect = max_indices[ii]
         local selectedPatches = batch:index(1, max_indices[ii]:sort():long())

         --local val,labels = max(tmp,1)
         --print("calculating x2 " .. i .. " " .. lasti)
         --print(selectedPatches:size())
         local x2 = sum(pow(selectedPatches,2),2)

         --print(x2)

         loss = loss + sum(x2*0.5 - val:t())

         -- count examplars per template
         local S = ii:t():float() -- zeros(m,k)
         --for i = 1,(#labels)[2] do
            -- S[i][labels[1][i]] = 1
         -- end
         --
         --print("printing sizes")
         --print(S:t():size())
         --print(selectedPatches:size())
         --print("end printing sizes")
         --print(summation:size())
         summation:add( S:t() * selectedPatches)
         counts:add( sum(S,1) )
      end

      -- normalize
      for i = 1,k do
         if counts[i] ~= 0 then
            centroids[i] = summation[i]:div(counts[i])
         elseif counts[i] == 0 then
            centroids[i] = centroids[i]*0
         end
      end

      -- total counts
      totalcounts:add(counts)

      -- callback?
      if callback then 
         local ret = callback(i,centroids:reshape(k_size),totalcounts) 
         if ret then break end
      end
   end

   return centroids,totalcounts
end
