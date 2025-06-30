import torch
import clip
import wandb
from PIL import Image
import torch.nn as nn
from utils import LT

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)  # CLIP.png为本文中图一，即CLIP的流程图
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)  # 将这三句话向量化
#
# with torch.no_grad():
#     # image_features = model.encode_image(image) # 将图片进行编码
#     # text_features = model.encode_text(text)    # 将文本进行编码
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]] # 图片"CLIP.png"对应"a diagram"的概率为0.9927937

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import LT

#我们的基础task seq2seq
#           1    2   3   4   5   6    7  8  9   10   11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
idx_char = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," "]
x= [7,4,11,11,14]
y= [7,4,11,11,14,26,22,14,17,11,3]
while len(x) < len(y):
    x.append(26)



batch_size = 1#构造h0的时候才需要
num_class = len(idx_char)
seq_len = len(x)
input_size = len(idx_char)  # 输入词汇表大小
output_size = len(idx_char) # 输出词汇表大小
num_layers = 2 #transform的层数

nhead = 1
hidden_size = 4 * nhead
seq_len_x = len(x)
seq_len_y = len(y)
embedding_size = nhead * 8 #数据嵌入层
inputs = torch.LongTensor(x).view(batch_size, seq_len)
ans = torch.LongTensor(y[:-1]).unsqueeze(0)
labels = torch.LongTensor(y).unsqueeze(0)


model = LT(input_size, output_size, hidden_size,
           num_layers, nhead,
           seq_len_x, seq_len_y,
           embedding_size,num_class,batch_size)

device = torch.device("cuda")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)

def tarin(inputs,ans,lables):
    inputs,labels,ans = inputs.to(device),lables.to(device),ans.to(device)
    for epoch in range(600):
        optimizer.zero_grad()#记得清零
        #开始训练网络
        locals = model(inputs,ans)
        loss = criterion(locals.view(-1, num_class), labels.view(-1)) #batch_size * seq_len,num_calss
        loss.backward()#所有的求和后反向传递
        optimizer.step()#参数更新
        if epoch % 10 == 0:
            _,idx = locals.max(-1)
            idx = idx.cpu()
            idx = idx.data.numpy()
            print("".join([idx_char[x] for x in idx[0]]), end ="")
            wandb.log({"loss": loss.item()})
            print(' ,Epoch [%d], Loss: %.6f' % (epoch, loss.item()))

wandb.init(project="lstm+tf", name="xlstm+tf")
tarin(inputs,ans,labels)
wandb.finish()