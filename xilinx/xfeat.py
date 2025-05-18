import os
import torch
import numpy as np
import torch.nn.functional as F
from vart_infer import VartInfer
from fine_matcher import FineMatcher

class XFeat():
    def __init__(self, 
                 first_model_path = 'model_data/first.xmodel',
                 second_model_path = 'model_data/second.xmodel',
                 fine_matcher_path = "model_data/xfeat.pt",
                 top_k = 4096, 
                 detection_threshold=0.05):
        super().__init__()
        self.dev = torch.device('cpu')
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        
        self.first_net = VartInfer(first_model_path)
        self.second_net = VartInfer(second_model_path)
        self.load_fine_matcher(fine_matcher_path)
        
        self.norm = torch.nn.InstanceNorm2d(1)
        
        self.first_size = (512, 704)  # h,w
        self.second_size = (1152, 1536)  # h,w
    
    def load_fine_matcher(self, fine_matcher_path):
        self.fine_matcher = FineMatcher().to(self.dev).eval()
        
        if not os.path.exists(fine_matcher_path):
            raise FileNotFoundError("model path is not exists.") 

        if isinstance(fine_matcher_path, str):
            print('loading weights from: ' + fine_matcher_path)
            self.fine_matcher.load_state_dict(torch.load(fine_matcher_path, map_location=self.dev), strict=False)
        else:
            self.fine_matcher.load_state_dict(fine_matcher_path, strict=False)
    
    def parse_input(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0,3,1,2)/255

        return x    
    
    def preprocess_tensor_quant(self, x, is_first = True):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = torch.tensor(x).permute(2,0,1)[None]
            elif len(x.shape) == 2:
                x = torch.tensor(x[..., None]).permute(2,0,1)[None]
            else:
                raise RuntimeError('For numpy arrays, only (H,W) or (H,W,C) format is supported.')
        
        
        if len(x.shape) != 4:
            raise RuntimeError('Input tensor needs to be in (B,C,H,W) format')
    
        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        # _H, _W = (H//32) * 32, (W//32) * 32
        if is_first:
            _H, _W = self.first_size
        else:
            _H, _W = self.second_size
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        x = x.mean(dim=1, keepdim = True)
        x = self.norm(x)

        return x, rh, rw
    
    def unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device = dev), 
                                torch.arange(w, device = dev), indexing='ij')
        xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
        return xy

    def extractDense(self, x, top_k = 8_000, is_first = True):
        if top_k < 1:
            top_k = 100_000_000

        if is_first:
            x, rh1, rw1 = self.preprocess_tensor_quant(x, is_first = True)
            unfold2d_input = self.unfold2d(x, ws=8)
            
            x = x.permute(0, 2, 3, 1).numpy()
            unfold2d_input = unfold2d_input.permute(0, 2, 3, 1).numpy()
            
            M1, H1 = self.first_net.Inference(x, unfold2d_input)
        else:
            x, rh1, rw1 = self.preprocess_tensor_quant(x, is_first = False)
            unfold2d_input = self.unfold2d(x, ws=8)
            
            x = x.permute(0, 2, 3, 1).numpy()
            unfold2d_input = unfold2d_input.permute(0, 2, 3, 1).numpy()
            
            M1, H1 = self.second_net.Inference(x, unfold2d_input)

        H1 = torch.nn.Sigmoid(H1)
        
        B, C, _H1, _W1 = M1.shape
        
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1)

        M1 = M1.permute(0,2,3,1).reshape(B, -1, C)
        H1 = H1.permute(0,2,3,1).reshape(B, -1)

        _, top_k = torch.topk(H1, k = min(len(H1[0]), top_k), dim=-1)

        feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64))
        mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)

        return mkpts, feats

    def extract_dualscale(self, x, top_k, s1 = 0.6, s2 = 1.3):
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

        B, _, _, _ = x.shape

        mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20), is_first = True)
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80), is_first = False)

        mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
        sc = torch.cat([sc1, sc2],dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, sc, feats

    def detectAndComputeDense(self, x, top_k = None, multiscale = True):
        """
            Compute dense *and coarse* descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return: features sorted by their reliability score -- from most to least
                List[Dict]: 
                    'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
                    'scales'       ->   torch.Tensor(top_k,): extraction scale
                    'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        """
        if top_k is None: top_k = self.top_k
        if multiscale:
            mkpts, sc, feats = self.extract_dualscale(x, top_k)
        else:
            mkpts, feats = self.extractDense(x, top_k)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc }

    def batch_match(self, feats1, feats2, min_cossim = -1):
        B = len(feats1)
        cossim = torch.bmm(feats1, feats2.permute(0,2,1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

        idx0 = torch.arange(len(match12[0]), device=match12.device)

        batched_matches = []

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0

            if min_cossim > 0:
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches

    def subpix_softmax2d(self, heatmaps, temp = 3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
        x = x - (W//2)
        y = y - (H//2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
        coords = coords.sum(1)

        return coords

    def refine_matches(self, d0, d1, matches, batch_idx, fine_conf = 0.25):
        idx0, idx1 = matches[batch_idx]
        feats1 = d0['descriptors'][batch_idx][idx0]
        feats2 = d1['descriptors'][batch_idx][idx1]
        mkpts_0 = d0['keypoints'][batch_idx][idx0]
        mkpts_1 = d1['keypoints'][batch_idx][idx1]
        sc0 = d0['scales'][batch_idx][idx0]

        #Compute fine offsets
        offsets = self.fine_matcher(torch.cat([feats1, feats2],dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1,8,8))

        mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        return torch.cat([mkpts_0, mkpts_1], dim=-1)

    def match_xfeat_star(self, im_set1, im_set2, top_k = None):
        """
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""
        if top_k is None: top_k = self.top_k
        im_set1 = self.parse_input(im_set1)
        im_set2 = self.parse_input(im_set2)

        #Compute coarse feats
        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

        #Match batches of pairs
        idxs_list = self.batch_match(out1['descriptors'], out2['descriptors'] )
        B = len(im_set1)

        #Refine coarse matches
        #this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(self.refine_matches(out1, out2, matches = idxs_list, batch_idx=b))

        return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())    