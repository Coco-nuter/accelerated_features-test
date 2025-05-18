import torch.nn as nn

class FineMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        
        ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher =  nn.Sequential(
                                            nn.Linear(128, 512),
                                            nn.BatchNorm1d(512, affine=True),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=True),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=True),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=True),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 64),
                                        )
        
    def forward(self, x):
        
        return self.fine_matcher(x)