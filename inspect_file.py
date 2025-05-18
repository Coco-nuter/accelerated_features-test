import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, Inspector
import time

from modules.model_v1_old import XFeatModel

DIVIDER = '-----------------------------------------'




def quantize(build_dir,quant_mode,inspect):

  quant_model = build_dir + '/quant_model'
  inspect_result = build_dir + '/inspect_result'
  float_model = build_dir + '/float_model'

  # use GPU if available   
  if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
      print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting device 0..')
    device = torch.device('cuda:0')
  else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

  input_first = torch.randn([1, 1, 512, 704], device=device)
  unfold2d_first = torch.randn([1, 64, 64, 88], device=device)
  model = XFeatModel().to(device).eval()
  if quant_mode == 'inspect':
      quant_model = model
      if inspect:
          target = "DPUCZDX8G_ISA1_B4096"
          inspector = Inspector(target)
          inspector.inspect(quant_model, (input_first, unfold2d_first), device=device, output_dir=inspect_result, image_format="png")
          sys.exit()
  # # load trained model
  # model = CNN().to(device)
  # model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))
  
  # quantizer = torch_quantizer(quant_mode, model, input, output_dir=quant_model)
  # quantized_model = quantizer.quant_model

  # # # evaluate 
  # # print("quantized_model start")
  # # start1 = time.time()
  # # test(quantized_model, device, test_loader)
  # # print("quantized_model time: ",time.time()-start1)
  # # print("original_model start")
  # # start2 = time.time()
  # # test(model, device, test_loader)
  # # print("original_model time: ",time.time()-start2)


  # # export config
  # if quant_mode == 'calib':
  #   quantizer.export_quant_config()
  # if quant_mode == 'test':
  #   quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return


def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test','inspect'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('--inspect', type=int, default=1 ,help='inspect model')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print('--inspect    : ', args.inspect)
  print(DIVIDER)

  quantize(args.build_dir, args.quant_mode, args.inspect)

  return



if __name__ == '__main__':
    run_main()

