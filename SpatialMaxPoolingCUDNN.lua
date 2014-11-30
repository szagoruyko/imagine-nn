local SpatialMaxPooling, parent = torch.class('imnn.SpatialMaxPooling', 'imnn._Pooling')

function SpatialMaxPooling:__init(kW, kH, dW, dH)
   parent.__init(self, kW, kH, dW, dH)
   self.mode = 'CUDNN_POOLING_MAX'
end

