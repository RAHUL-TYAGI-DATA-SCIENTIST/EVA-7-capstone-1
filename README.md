# EVA-7-Capstone-1
![Test Image 1](https://user-images.githubusercontent.com/70502759/158041859-474702f1-b5e0-4498-abe7-34685cdd8183.png)

Below are the steps performed in DETR for Panoptic segmentation :
Assuming we have sample image of batch 2 with shape : [batch = 2 , channel (C) = 3 , hight (H) = 873 , width (W) = 1060 ]

## Move 1 : BackBone in DETR architecture

  we know that the backbone upon accepting an input of above shape  returns output and encoding, 
  where output has intermediate layer  tensors  and positional encoding of following shape 

     layer 0 :  tensors shape : [batch = 2, channel = 256, hight = 219, width = 265] 
                mask shape    : [batch = 2,  hight = 219, width = 265 ]
	        pos encoding  : [batch = 2, channel = 256, hight = 219, width = 265] 
	   

     layer 1 :  tensors shape : [batch = 2, channel = 512, hight = 110, width = 113] 
                mask shape    : [batch = 2,  hight = 110, width = 113 ]
	        pos encoding  : [batch = 2, channel = 256, hight = 110, width = 113] 
	   
     layer 2 :  tensors shape : [batch = 2, channel = 1024, hight = 55, width = 67] 
                mask shape    : [batch = 2,  hight = 55, width = 67 ]
                    pos encoding  : [batch = 2, channel = 256, hight = 55, width = 67] 

     layer 3 :  tensors shape : [batch = 2, channel = 2048, hight = 28, width = 34] 
                mask shape    : [batch = 2,  hight = 28, width = 34 ]
	        pos encoding  : [batch = 2, channel = 256, hight = 28, width = 34] 


## Move 2 : We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention

  Backbone last layer o/p again pass throw convolution network and 
  project back in to  Size of the embeddings (dimension as required by the transformer)

  i/p  shape      : tensors shape : [batch = 2, channel = 2048, hight = 28, width = 34] 
  projected shape : tensors shape : [batch = 2, channel = 256, hight = 28, width = 34] 
  
  These project back embeddings along with positional embedding passed to transformer architecture .
  Transformer will return output from encoder as encoded image (called as memory ) 
  and N object queries are transformed into an output embedding by the decoder . 
  
  Hence encoded image is the output of transformer encoder which is of following shape and   is passed to Multi-Head Attention 
  tensors shape : [batch = 2, embedding  vector(d) :256, hight(H/32) = 28, width(N/32) = 34]



## Move 3 : We also send dxN Box embeddings to the Multi-Head Attention

     N object queries are transformed into an output embedding by the decoder  and 
     last decoder layer output is passed to Multi-Head Attention along with output of step 2 i.e encoded image.
     tensors shape : [batch = 2, object queries size(N) = 100, embedding  vector(d) : 256]

## Move 4 : We do something here to generate NxMxH/32xW/32 maps

Multi Head Attention Map will take input of transformer decoder last layer (from step 3) output and 
encoder encoded image output (from step 2) .
Transformer decoder last layer output behaves as query and encoder output behaves as key . 
Then we calculate a self-attention score. The score is calculated by taking the dot 
product of the query vector with the key vector. 
The score determines how much focus to place on other parts of the input image at a certain position 
and Multi Head Attention Map module only returns the attention softmax

  Details coding steps in Multi Head Attention Map : 
  
  
 def forward(self, q, k, mask: Optional[Tensor] = None):  # q = [2 100 256 ] k = [2 256 28 34]  mask = [2 28 34]
 
       q = self.q_linear(q)  # q will be projected
   
       k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)  # k will be projected
    
       qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)	   
       # qq reshape  based on number of heads to optimize the matrix multiplication qh = [2 100 8 32]
  
       kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
       # kh reshape  based on number of heads to optimize the matrix multiplication  kh [2 8 32 24 32]

       weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)  # self attention weight = [2 100 8 28 34 ]

       if mask is not None:
          weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

       weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())  # attention softmax 

       weights = self.dropout(weights)  # dropout

       return weights  # Batch N = 2 Object Query M = 100 No of Head= 8 Hight = 28 width = 34
       
 ![Test Image 2](https://user-images.githubusercontent.com/70502759/158050222-87865ae1-a803-4364-9115-c25a4753deba.PNG)


## Move 5 : Then we concatenate these maps with Res5 Block

Attention map from step 4 and resnet BackBone block from step 1 
will concatenate and will do upsampling  using a below FPN approach . 
 
### x = [2 256 28 34] bbox_mask = [2 100 8 28 34 ]
### fpns # 0 > 2 1024 48 64 # 1 > 2 512 96 128 # 2 > 2 256 192 256
def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):

    x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)],
                  1)  # after expanded x [ 200 256 28 34 ],  bbox_mask [ 200 8 28 34 ]
    # output x [  200 264 28 34 ]
    x = self.lay1(x)
### output x [  200 264 28 34 ]
    x = self.gn1(x)
    x = F.relu(x)

    x = self.lay2(x)
### output x [  200 128 28 34 ]

    x = self.gn2(x)
    x = F.relu(x)

    cur_fpn = self.adapter1(fpns[0])
    # op cur_fpn [ 2 128 55 67 ]
    if cur_fpn.size(0) != x.size(0):
        cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))  # expanded cur_fpn 200 128 55 67
    x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
    # cur_fpn 200 128 55 67   +  output x [  200 128 55 67 ]
    # output x [  200 128 55 67 ]
    x = self.lay3(x)
    # output x [  200 64 55 67 ]
    x = self.gn3(x)

    x = F.relu(x)

    cur_fpn = self.adapter2(fpns[1])  # output cur_fpn [ 2 64 110 133 ]
    if cur_fpn.size(0) != x.size(0):
        cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))  # expanded cur_fpn  [ 200 64 110 133 ]
    x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
    # output cur_fpn [  200 64 110 133] + x [ 200 64 110 133 ]
    x = self.lay4(x)
    # output x [  200 32 110 133 ]
    x = self.gn4(x)
    x = F.relu(x)

    cur_fpn = self.adapter3(fpns[2])  # output x [ 2 32 219 265 ]
    if cur_fpn.size(0) != x.size(0):
        cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))  # output cur_fpn [ 200 32 219 265 ]
    x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
    # output cur_fpn [  200 32 219 265] + x [ 200 32 219 265 ]
    x = self.lay5(x)  # output x [ 200 16 219 265 ]
    x = self.gn5(x)
    x = F.relu(x)

    x = self.out_lay(x)  # output x [ 200 1 219 265 ]
    return x  
### output x [ 200 1 192 265 ]  
### this will be reshaped latter into [batch 2 No of object query detection = 100 Hight(H/4) 192 Width(W/4) = 265]

![Test Image 3](https://user-images.githubusercontent.com/70502759/158050613-14e2af1e-822f-45e5-a1ce-38e4b8210b2c.PNG)

## Move 6 : Then we perform the below steps

      Detr will give below output after performing all above steps : 
  
      pred_boxes [2, 100, 4] -> This can detect 100 object bouding box coordinate details 
  
      pred_logits [2, 100, 251] -> This can detect max 100 object in the image
  
      pred_masks  [2, 100, 192, 265] -> This will hold 100 predicted object pixel wise segmented details 
  
     We filter the predictions for which the confidence is less then certain threshold .
     
 ![Test Image 4](https://user-images.githubusercontent.com/70502759/158048064-107b938c-a5bd-46c9-b326-dcedcb0e3e86.PNG)    

## Move 7 : Now we will go for panoptic segmentation

At the end, the remaining masks are merged together using a pixel-wise argmax . 


Questions & Answers:
1. Where do we take the encoded image from?
We obtain the encoded image from the output of the image encoder CNN from the object detection stage of DETR.

2. How do we generate attention maps?
(N x M x H/32 x W/32) attention maps are low resolution attention maps generated from the multi head attention attention module with M heads. It generates M low resolution heatmaps per N objects in the encoded image feature resolution of H/32 x W/32.

3. Where is the Res5 block coming from?
Res5, Res4, Res3, Res2 are feature blocks derived from the original image encoder CNN. In the case of the original paper, considering Resnet is used, these are Resnet features.

4. Explain the DETR Panoptic Segmentation steps.
Explanation for sequential steps of DETR Panoptic Segmentation Mask Head: Step 1: Input box embeddings into multi head attention module with M heads to calculate attention maps over encoded image. M attention maps are output for every object (over N objects) with the same encoded image feature resolution of H/32 x W/32 Step 2: Attention maps obtained are parallely upsampled using a FPN style CNN to create mask logits for every object. This requires input of ResNet features as well. Step 3: Upsampling CNN output contains mask logits for every object, on which pixelwise argmax operation can be performed to obtain the final panoptic segmented output.

Steps to Train DETR for Panoptic Segmentation
Train DETR object detection model to detect and ouput bounding boxes for both things (objects) and stuff (background objects) categories.
Freeze all weights and train mask head independently for fewer epochs.
