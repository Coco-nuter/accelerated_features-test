1. python -m modules.eval.megadepth1500 --dataset-dir MegaDepth-1500/Mega1500/  --matcher xfeat-star --ransac-thr 2.5 --quant
2. python -m modules.eval.megadepth1500 --dataset-dir MegaDepth-1500/Mega1500/  --matcher xfeat-star --ransac-thr 2.5 --quant --quant-mode test
3. vai_c_xir -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -x first_model/XFeatModel_int.xmodel -o first_model/ -n first
4. python -m modules.eval.megadepth1500 --dataset-dir MegaDepth-1500/Mega1500/  --matcher xfeat-star --ransac-thr 2.5 --quant --quant-s
5. python -m modules.eval.megadepth1500 --dataset-dir MegaDepth-1500/Mega1500/  --matcher xfeat-star --ransac-thr 2.5 --quant --quant-s --quant-mode test
4. vai_c_xir -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -x second_model/XFeatModel_int.xmodel -o second_model/ -n second
