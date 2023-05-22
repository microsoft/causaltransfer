# stdlib
from typing import Tuple

import ganite.logger as log

# third party
import numpy as np
import torch
from ganite.utils.random import enable_reproducible_results
from torch import nn
# from mmd import mmd_rbf
# from mmd_loss import MMD_loss
from itertools import combinations
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from tqdm import tqdm
from PIL import Image
from utils import compute_irm_penalty, compute_mmd
import pdb
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10

class CounterfactualGenerator(nn.Module):
    """
    The counterfactual generator, G, uses the feature vector x,
    the treatment vector t, and the factual outcome yf, to generate
    a potential outcome vector, hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int, binary_y: bool, DimOutput: int
    ) -> None:
        super(CounterfactualGenerator, self).__init__()
        # Generator Layer
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Dropout(),
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )
        self.common = nn.Sequential(
            nn.Linear(
                Dim + 1 + DimOutput, DimHidden
            ),  # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector      (Z)
            nn.LeakyReLU(),
            *hidden,
        ).to(DEVICE)

        self.binary_y = binary_y
        self.DimOutput = DimOutput
        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, DimOutput),
                ).to(DEVICE)
            )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        inputs = torch.cat([x, t, y], dim=1).to(DEVICE)

        G_h2 = self.common(inputs)

        G_prob1 = self.outs[0](G_h2)
        G_prob2 = self.outs[1](G_h2)

        G_prob = torch.cat([G_prob1, G_prob2], dim=1).to(DEVICE)

        if self.binary_y:
            return torch.sigmoid(G_prob)
        elif self.DimOutput > 1:
            G_pred1 = torch.softmax(G_prob1, dim=1)
            G_pred2 = torch.softmax(G_prob2, dim=1)
            G_pred = torch.cat([G_pred1, G_pred2], dim=1).to(DEVICE)
            return G_pred
        else:
            return G_prob

class CounterfactualDiscriminatorSeparateT(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(
        self, Dim: int, Treatments: list, DimHidden: int, depth: int, binary_y: bool, DimOutput: int
    ) -> None:
        super(CounterfactualDiscriminatorSeparateT, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.Treatments = Treatments

        self.model = nn.Sequential(
            nn.Linear(Dim + DimOutput + 1, DimHidden),
            nn.LeakyReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
    
        inputs = torch.cat([x, y, t], dim=1).to(DEVICE)
        return self.model(inputs)

class CounterfactualDiscriminatorSeparate(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(
        self, Dim: int, Treatments: list, DimHidden: int, depth: int, binary_y: bool, DimOutput: int
    ) -> None:
        super(CounterfactualDiscriminatorSeparate, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.Treatments = Treatments

        self.model = nn.Sequential(
            nn.Linear(Dim + DimOutput, DimHidden),
            nn.LeakyReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
    
        inputs = torch.cat([x, y], dim=1).to(DEVICE)
        return self.model(inputs)


class CounterfactualDiscriminator(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(
        self, Dim: int, Treatments: list, DimHidden: int, depth: int, binary_y: bool, DimOutput: int
    ) -> None:
        super(CounterfactualDiscriminator, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.Treatments = Treatments

        self.model = nn.Sequential(
            nn.Linear(Dim + len(Treatments)*DimOutput, DimHidden),
            nn.LeakyReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, hat_y: torch.Tensor
    ) -> torch.Tensor:
        # Factual & Counterfactual outcomes concatenate
        # TODO: Have to check dimensions here?
        inp0 = (1.0 - t) * y + t * hat_y[:, 0].reshape([-1, 1])
        inp1 = t * y + (1.0 - t) * hat_y[:, 1].reshape([-1, 1])

        inputs = torch.cat([x, inp0, inp1], dim=1).to(DEVICE)
        return self.model(inputs)


class InferenceNets(nn.Module):
    """
    The ITE generator uses only the feature vector, x, to generate a potential outcome vector hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int, binary_y: bool, DimOutput: int
    ) -> None:
        super(InferenceNets, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.common = nn.Sequential(
            nn.Linear(Dim, DimHidden),
            nn.LeakyReLU(),
            *hidden,
        ).to(DEVICE)
        self.binary_y = binary_y
        self.DimOutput = DimOutput

        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, DimOutput),
                ).to(DEVICE)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        I_h = self.common(x)

        I_probs = []
        for out in self.outs:
            I_probs.append(out(I_h))

        if self.binary_y:
            return torch.sigmoid(torch.cat(I_probs, dim=1).to(DEVICE))
        else:
            return torch.cat(I_probs, dim=1).to(DEVICE)


class CounterfactualGAN(nn.Module):

    def __init__(
        self,
        X: torch.Tensor,
        Treatments: torch.Tensor,
        Y: torch.Tensor,
        clusters: torch.Tensor,
        data_dir,
        task, 
        X_text: np.ndarray = None,
        X_image: np.ndarray = None,
        language_model: str = None,
        image_model: str = None,
        dim_hidden: int = 100,
        alpha: float = 0.1,
        beta: float = 0,
        minibatch_size: int = 32,
        depth: int = 0,
        num_iterations: int = 5000,
        num_discr_iterations: int = 1,
        IRM: bool = False,
        IRM_generator: bool = False,
        counterfactual_invariance = False,
        counterfactual_invariance_generator = False,
        separate_discriminators = False,
        discriminator_t = True,
        lm_gen = False,
        save = True,
        save_dir: str = './',
        num_selftrain_iterations: int=1,
        share_weights: bool = False
    ) -> None:
        super(CounterfactualGAN, self).__init__()

        X = self._check_tensor(X)
        Treatments = self._check_tensor(Treatments)
        Y = self._check_tensor(Y)

        if np.isnan(np.sum(X.cpu().numpy())):
            raise ValueError("X contains NaNs")
        if len(X) != len(Treatments):
            raise ValueError("Features/Treatments mismatch")
        if len(X) != len(Y):
            raise ValueError("Features/Labels mismatch")

        enable_reproducible_results()

        dim_in = X.shape[1]
        self.original_treatments = np.sort(np.unique(Treatments.cpu().numpy()))
        self.treatments = [0, 1]

        if len(self.original_treatments) != 2:
            raise ValueError("Only two treatment categories supported")

        # Hyperparameters
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.num_iterations = num_iterations
        self.num_discr_iterations = num_discr_iterations
        self.IRM = IRM
        self.IRM_generator = IRM_generator
        self.counterfactual_invariance = counterfactual_invariance
        self.counterfactual_invariance_generator = counterfactual_invariance_generator
        self.lm_gen = lm_gen
        self.num_selftrain_iterations = num_selftrain_iterations
        self.share_weights = share_weights
        self.data_dir = data_dir
        self.separate_discriminators = separate_discriminators
        self.discriminator_t = discriminator_t

        dim_output = 1
        binary_y = len(np.unique(Y.cpu().numpy())) == 2
        if task == 'multiclass':
            dim_output = len(np.unique(Y.cpu().numpy()))

        if not X_text is None:
            self.language_model = AutoModel.from_pretrained(language_model)
            self.tokenizer = AutoTokenizer.from_pretrained(language_model, max_length=512, padding=True, truncation=True)
            self.language_model.to(DEVICE)
        
        if not X_image is None:
            self.image_model = AutoModel.from_pretrained(image_model)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(image_model)
            self.image_model.to(DEVICE)

        if self.lm_gen:
            if X_text is not None:
                dim_in += self.language_model.config.dim
            if X_image is not None:
                dim_in += self.image_model.config.hidden_size
        # Layers
        self.counterfactual_generator = CounterfactualGenerator(
            dim_in, len(self.treatments), dim_hidden, depth, binary_y, dim_output
        ).to(DEVICE)
        if not self.separate_discriminators:
            self.counterfactual_discriminator = CounterfactualDiscriminator(
                dim_in, self.treatments, dim_hidden, depth, binary_y, dim_output
            ).to(DEVICE)
            self.DG_solver = torch.optim.Adam(
                list(self.counterfactual_generator.parameters()) + list(self.counterfactual_discriminator.parameters()),
                lr=1e-3,
                eps=1e-8,
                weight_decay=1e-3,
            )
        else:
            if self.discriminator_t:
                self.counterfactual_discriminator0 = CounterfactualDiscriminatorSeparateT(
                    dim_in, self.treatments, dim_hidden, depth, binary_y, dim_output
                ).to(DEVICE)
                self.counterfactual_discriminator1 = CounterfactualDiscriminatorSeparateT(
                    dim_in, self.treatments, dim_hidden, depth, binary_y, dim_output
                ).to(DEVICE)
            else:
                self.counterfactual_discriminator0 = CounterfactualDiscriminatorSeparate(
                    dim_in, self.treatments, dim_hidden, depth, binary_y, dim_output
                ).to(DEVICE)
                self.counterfactual_discriminator1 = CounterfactualDiscriminatorSeparate(
                    dim_in, self.treatments, dim_hidden, depth, binary_y, dim_output
                ).to(DEVICE)
            self.DG_solver = torch.optim.Adam(
                list(self.counterfactual_generator.parameters()) + 
                list(self.counterfactual_discriminator0.parameters()) +
                list(self.counterfactual_discriminator1.parameters()),
                lr=1e-3,
                eps=1e-8,
                weight_decay=1e-3,
            )
        self.inference_nets = InferenceNets(
            dim_in, len(self.treatments), dim_hidden, depth, binary_y, dim_output
        ).to(DEVICE)

        self.I_solver = torch.optim.Adam(
            self.inference_nets.parameters(), lr=1e-3, weight_decay=1e-3
        )

        if self.lm_gen:
            if X_text is not None:
                self.DG_lm_solver = torch.optim.Adam(
                    self.language_model.parameters(),
                    lr=2e-5
                )
            if X_image is not None:
                self.DG_vm_solver = torch.optim.Adam(
                    self.image_model.parameters(),
                    lr=2e-5
                )
        
        if X_text is not None:
            self.I_lm_solver = torch.optim.Adam(
                self.language_model.parameters(),
                lr=2e-5
            )
        if X_image is not None:
            self.I_vm_solver = torch.optim.Adam(
                self.image_model.parameters(),
                lr=2e-5
            )

        save_path = os.path.join(save_dir, 'counterfactual_gan.pt')

        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.DG_solver.load_state_dict(checkpoint['DG_solver_state_dict'])
            self.I_solver.load_state_dict(checkpoint['I_solver_state_dict'])
            if self.lm_gen:
                if X_text is not None:
                    self.DG_lm_solver.load_state_dict(checkpoint['DG_lm_solver_state_dict'])
                if X_image is not None:
                    self.DG_vm_solver.load_state_dict(checkpoint['DG_vm_solver_state_dict'])
            if X_text is not None:
                self.I_lm_solver.load_state_dict(checkpoint['I_lm_solver_state_dict'])
            if X_image is not None:
                self.I_vm_solver.load_state_dict(checkpoint['I_vm_solver_state_dict'])

        else:
            self._fit(X, Treatments, Y, clusters, dim_output, task, X_text, X_image)
            if save:
                print('Saving CounterfactualGAN parameters')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, 'counterfactual_gan.pt')
                save_dict = {
                    'model_state_dict': self.state_dict(),
                    'DG_solver_state_dict': self.DG_solver.state_dict(),
                    'I_solver_state_dict': self.I_solver.state_dict(),
                }
                if self.lm_gen:
                    if X_text is not None:
                        save_dict['DG_lm_solver_state_dict'] = self.DG_lm_solver.state_dict()
                    if X_image is not None:
                        save_dict['DG_vm_solver_state_dict'] = self.DG_vm_solver.state_dict()
                if X_text is not None:
                    save_dict['I_lm_solver_state_dict'] = self.I_lm_solver.state_dict()
                if X_image is not None:
                    save_dict['I_vm_solver_state_dict'] = self.I_vm_solver.state_dict()
                    
                torch.save(save_dict, save_path)

    # def _sample_minibatch(
    #     self, X: torch.Tensor, T: torch.tensor, Y: torch.Tensor, clusters: torch.Tensor
    # ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.Tensor]:
    def _sample_minibatch(
        self, X: torch.Tensor, X_text, X_image, T: torch.tensor, Y: torch.Tensor, clusters: torch.Tensor, gen_embeds: bool, dim_output: int):
        idx_mb = np.random.randint(0, X.shape[0], self.minibatch_size)
        X_mb = X[idx_mb, :]
        T_mb = T[idx_mb].reshape([self.minibatch_size, 1])
        Y_mb = Y[idx_mb].reshape([self.minibatch_size, dim_output])
        cluster_mb = clusters[idx_mb]

        if not gen_embeds:
            return X_mb, T_mb, Y_mb, cluster_mb

        if X_image is not None:
            X_image_mb = X_image[idx_mb]
            img_list = [Image.open(os.path.join(self.data_dir, 'airbnb/images/{}'.format(filepath.split('/')[-1]))).convert(mode='RGB')
                for filepath in X_image_mb]
            X_image_mb = self.feature_extractor(images=img_list, return_tensors='pt')
            for img in img_list:
                img.close()
                del(img)

        if X_text is not None:
            X_text_mb = X_text[idx_mb]
            X_text_mb = self.tokenizer(X_text_mb.tolist(), padding='max_length', truncation=True, return_tensors='pt')
            
        if X_text is not None and X_image is not None:
            return X_mb, X_text_mb, X_image_mb, T_mb, Y_mb, cluster_mb
        elif X_text is not None and X_image is None:
            return X_mb, X_text_mb, T_mb, Y_mb, cluster_mb
        elif X_text is None and X_image is not None:
            return X_mb, X_image_mb, T_mb, Y_mb, cluster_mb
        
        return X_mb, T_mb, Y_mb, cluster_mb

    def _process_X_text(self, X_mb, X_text, eval=False): # DEFAULT ONE
        return X_mb

    def _process_X_image(self, X_mb, X_image, eval=False):
        return X_mb

    def _sample_outcomes(self, T_mb, Y_mb, Tilde):
        y0 = Y_mb[(T_mb == 0).flatten()]
        y1 = Y_mb[(T_mb == 1).flatten()]
        half_tilde = int(Tilde.shape[1]/2)
        y0_hat = Tilde[:,:half_tilde]
        y1_hat = Tilde[:,half_tilde:]
        # y0_hat = Tilde[:,0]
        # y1_hat = Tilde[:,1]

        half_ceil = int(np.ceil(Y_mb.shape[0]/2))
        half_floor = int(np.floor(Y_mb.shape[0]/2))

        if len(y0) < int(Y_mb.shape[0]/2):
            y0_idxs = torch.argsort(torch.rand(Y_mb.shape[0]))
            y0_mix = torch.vstack([y0, y0_hat[y0_idxs][:int(Y_mb.shape[0]-len(y0))]])
            cf0_mix = torch.vstack([torch.zeros(len(y0)).view(-1, 1), torch.ones(Y_mb.shape[0]-len(y0)).view(-1, 1)])
            shuffle_idxs0 = torch.argsort(torch.rand(Y_mb.shape[0]))
            y0_mix = y0_mix[shuffle_idxs0]
            cf0_mix = cf0_mix[shuffle_idxs0]

            y1_real_idxs = torch.argsort(torch.rand(y1.shape[0]))
            y1_cf_idxs = torch.argsort(torch.rand(y1_hat.shape[0]))
            y1_mix = torch.vstack([y1[y1_real_idxs][:half_ceil], y1_hat[y1_cf_idxs][half_ceil:]])
            cf1_mix = torch.vstack([torch.zeros(half_ceil).view(-1, 1), torch.ones(half_floor).view(-1, 1)])
            shuffle_idxs1 = torch.argsort(torch.rand(Y_mb.shape[0]))
            y1_mix = y1_mix[shuffle_idxs1]
            cf1_mix = cf1_mix[shuffle_idxs1]

        elif len(y1) < Y_mb.shape[0]/2:
            y1_idxs = torch.argsort(torch.rand(Y_mb.shape[0]))
            y1_mix = torch.vstack([y1, y1_hat[y1_idxs][:int(Y_mb.shape[0]-len(y1))]])
            cf1_mix = torch.vstack([torch.zeros(len(y1)).view(-1, 1), torch.ones(Y_mb.shape[0]-len(y1)).view(-1, 1)])
            shuffle_idxs1 = torch.argsort(torch.rand(Y_mb.shape[0]))
            y1_mix = y1_mix[shuffle_idxs1]
            cf1_mix = cf1_mix[shuffle_idxs1]

            y0_real_idxs = torch.argsort(torch.rand(y0.shape[0]))
            y0_cf_idxs = torch.argsort(torch.rand(y0_hat.shape[0]))
            y0_mix = torch.vstack([y0[y0_real_idxs][:half_ceil], y0_hat[y0_cf_idxs][half_ceil:]])
            cf0_mix = torch.vstack([torch.zeros(half_ceil).view(-1, 1), torch.ones(half_floor).view(-1, 1)])
            shuffle_idxs0 = torch.argsort(torch.rand(Y_mb.shape[0]))
            y0_mix = y0_mix[shuffle_idxs0]
            cf0_mix = cf0_mix[shuffle_idxs0]
        
        else:

            y0_real_idxs = torch.argsort(torch.rand(y0.shape[0]))
            y0_cf_idxs = torch.argsort(torch.rand(y0_hat.shape[0]))
            y0_mix = torch.vstack([y0[y0_real_idxs][:half_ceil], y0_hat[y0_cf_idxs][half_ceil:]])
            cf0_mix = torch.vstack([torch.zeros(half_ceil).view(-1, 1), torch.ones(half_floor).view(-1, 1)])
            shuffle_idxs0 = torch.argsort(torch.rand(Y_mb.shape[0]))
            y0_mix = y0_mix[shuffle_idxs0]
            cf0_mix = cf0_mix[shuffle_idxs0]
        
            y1_real_idxs = torch.argsort(torch.rand(y1.shape[0]))
            y1_cf_idxs = torch.argsort(torch.rand(y1_hat.shape[0]))
            y1_mix = torch.vstack([y1[y1_real_idxs][:half_ceil], y1_hat[y1_cf_idxs][half_ceil:]])
            cf1_mix = torch.vstack([torch.zeros(half_ceil).view(-1, 1), torch.ones(half_floor).view(-1, 1)])
            shuffle_idxs1 = torch.argsort(torch.rand(Y_mb.shape[0]))
            y1_mix = y1_mix[shuffle_idxs1]
            cf1_mix = cf1_mix[shuffle_idxs1]
        
        y0_mix = y0_mix.to(DEVICE)
        cf0_mix = cf0_mix.to(DEVICE)
        y1_mix = y1_mix.to(DEVICE)
        cf1_mix = cf1_mix.to(DEVICE)

        return (y0_mix, cf0_mix, y1_mix, cf1_mix)

    def _fit(
        self,
        X: torch.Tensor,
        Treatment: torch.Tensor,
        Y: torch.Tensor,
        clusters: torch.Tensor,
        dim_output: int,
        task,
        X_text: np.ndarray = None,
        X_image: np.ndarray = None,
    ) -> "CounterfactualGAN":
        Train_X = self._check_tensor(X).float()
        Train_T = self._check_tensor(Treatment).float().reshape([-1, 1])
        Train_Y = self._check_tensor(Y).float().reshape([-1, 1])
        if task == 'multiclass':
            Train_Y = torch.nn.functional.one_hot(Train_Y.flatten().long(), num_classes=dim_output).to(torch.float32)

        # Encode
        min_t_val = Train_T.min()
        Train_T = (Train_T > min_t_val).float()

        # Iterations
        # Train G and D first
        self.counterfactual_generator.train()
        if not self.separate_discriminators:
            self.counterfactual_discriminator.train()
        else:
            self.counterfactual_discriminator0.train()
            self.counterfactual_discriminator1.train()
        self.inference_nets.train()
        if X_text is not None:
            self.language_model.train()
        if X_image is not None:
            self.image_model.train()

        # if self.IRM_generator:
        #     dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE)

        print ('Training G and D')
        for it in tqdm(range(self.num_iterations)):
            self.DG_solver.zero_grad()
            if self.lm_gen:
                if X_text is not None:
                    self.DG_lm_solver.zero_grad()
                if X_image is not None:
                    self.DG_vm_solver.zero_grad()

            # penalty_multiplier = 2.0 * it / self.num_iterations ** 2.0
            # if self.IRM_generator:
            penalty_multiplier = 2.0 * it / self.num_iterations ** 1.6
            # elif self.counterfactual_invariance_generator:
            #     penalty_multiplier = 2.0 * it / self.num_iterations ** 3

            if self.IRM_generator:                
                error = 0
                penalty = 0
            if not self.lm_gen or (X_text is None and X_image is None):
                X_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=False, dim_output=dim_output)
                X_text_mb = None
                X_image_mb = None
            elif X_text is not None and X_image is not None:
                X_mb, X_text_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
            elif X_text is not None and X_image is None:
                X_mb, X_text_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                X_image_mb = None
            elif X_text is None and X_image is not None:
                X_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                X_text_mb = None
            
            for kk in range(self.num_discr_iterations):
                if self.lm_gen:
                    X_mb_concat = self._process_X_text(X_mb, X_text_mb)
                    X_mb_concat = self._process_X_image(X_mb_concat, X_image_mb)
                else:
                    X_mb_concat = X_mb

                self.DG_solver.zero_grad()
                if self.lm_gen:
                    if X_text is not None:
                        self.DG_lm_solver.zero_grad()
                    if X_image is not None:
                        self.DG_vm_solver.zero_grad()

                # For multiclass, need to re-encode Y as a one-hot matrix of 32x5
                Tilde = self.counterfactual_generator(X_mb_concat, T_mb, Y_mb).clone()
                if not self.separate_discriminators:
                    D_out = self.counterfactual_discriminator(X_mb_concat, T_mb, Y_mb, Tilde)

                    if torch.isnan(Tilde).any():
                        raise RuntimeError("counterfactual_generator generated NaNs")
                    if torch.isnan(D_out).any():
                        raise RuntimeError("counterfactual_discriminator generated NaNs")

                    D_loss = nn.BCELoss()(D_out, T_mb)
                
                else:

                    y0_mix, cf0_mix, y1_mix, cf1_mix = self._sample_outcomes(T_mb, Y_mb, Tilde)

                    if self.discriminator_t:
                        D0_out = self.counterfactual_discriminator0(X_mb_concat, y0_mix, torch.zeros(Y_mb.shape[0]).view(-1, 1).to(DEVICE))
                        D1_out = self.counterfactual_discriminator1(X_mb_concat, y1_mix, torch.ones(Y_mb.shape[0]).view(-1, 1).to(DEVICE))
                    else:
                        D0_out = self.counterfactual_discriminator0(X_mb_concat, y0_mix)
                        D1_out = self.counterfactual_discriminator1(X_mb_concat, y1_mix)

                    if torch.isnan(Tilde).any():
                        raise RuntimeError("counterfactual_generator generated NaNs")
                    if torch.isnan(D0_out).any() or torch.isnan(D1_out).any():
                        raise RuntimeError("counterfactual_discriminator generated NaNs")

                    D0_loss = nn.BCELoss()(D0_out, cf0_mix)
                    D1_loss = nn.BCELoss()(D1_out, cf1_mix)
                    D_loss = D0_loss + D1_loss

                D_loss.backward()

                self.DG_solver.step()
                if self.lm_gen:
                    if X_text is not None:
                        self.DG_lm_solver.step()
                    if X_image is not None:
                        self.DG_vm_solver.step()

            if self.lm_gen:
                X_mb_concat = self._process_X_text(X_mb, X_text_mb)
                X_mb_concat = self._process_X_image(X_mb_concat, X_image_mb)
            else:
                X_mb_concat = X_mb
            Tilde = self.counterfactual_generator(X_mb_concat, T_mb, Y_mb)
            if not self.separate_discriminators:
                D_out = self.counterfactual_discriminator(X_mb_concat, T_mb, Y_mb, Tilde)
                D_loss = nn.BCELoss()(D_out, T_mb)
            
            else:
            
                y0_mix, cf0_mix, y1_mix, cf1_mix = self._sample_outcomes(T_mb, Y_mb, Tilde)
                
                if self.discriminator_t:
                    D0_out = self.counterfactual_discriminator0(X_mb_concat, y0_mix, torch.zeros(Y_mb.shape[0]).view(-1, 1).to(DEVICE))
                    D1_out = self.counterfactual_discriminator1(X_mb_concat, y1_mix, torch.ones(Y_mb.shape[0]).view(-1, 1).to(DEVICE))
                else:
                    D0_out = self.counterfactual_discriminator0(X_mb_concat, y0_mix)
                    D1_out = self.counterfactual_discriminator1(X_mb_concat, y1_mix)

                D0_loss = nn.BCELoss()(D0_out, cf0_mix)
                D1_loss = nn.BCELoss()(D1_out, cf1_mix)
                D_loss = D0_loss + D1_loss

            G_loss_GAN = D_loss

            if self.IRM_generator:
                unique_clusters = list(set(cluster_mb))
                for cluster in unique_clusters:
                    dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE).requires_grad_()
                    T_cluster = T_mb[cluster_mb == cluster, :]
                    Y_cluster = Y_mb[cluster_mb == cluster, :]
                    Tilde_cluster = Tilde[cluster_mb == cluster, :]
                
                    if task == 'classification' or task == 'regression':
                        G_loss_R = nn.MSELoss()(
                                Y_cluster,
                                (T_cluster * Tilde_cluster[:, 1].reshape([-1, 1])
                                + (1.0 - T_cluster) * Tilde_cluster[:, 0].reshape([-1, 1])) * dummy_w,
                            )
                    elif task == 'multiclass':
                        G_loss_R = nn.MSELoss()(
                                Y_cluster,
                                (T_cluster * Tilde_cluster[:, 5:]
                                + (1.0 - T_cluster) * Tilde_cluster[:, 0:5]) * dummy_w,
                            )

                    G_loss = G_loss_R + self.alpha * G_loss_GAN
                    penalty += compute_irm_penalty(G_loss, dummy_w)
                    error += G_loss.mean()

                if it % 100 == 0:
                    log.debug(f"Generator loss epoch {it}: {D_loss} {G_loss}")
                    if torch.isnan(D_loss).any():
                        raise RuntimeError("counterfactual_discriminator generated NaNs")

                    if torch.isnan(G_loss).any():
                        raise RuntimeError("counterfactual_generator generated NaNs")
                
                if penalty_multiplier > 1.0:
                    error /= penalty_multiplier

                (error + penalty_multiplier * penalty).backward()

            else:
                if task == 'classification' or task == 'regression':
                    G_loss_R = torch.mean(
                        nn.MSELoss()(
                            Y_mb,
                            T_mb * Tilde[:, 1].reshape([-1, 1])
                            + (1.0 - T_mb) * Tilde[:, 0].reshape([-1, 1]),
                        )
                    )
                elif task == 'multiclass':
                    G_loss_R = torch.mean(
                        nn.MSELoss()(
                            Y_mb,
                            T_mb * Tilde[:, 5:]
                            + (1.0 - T_mb) * Tilde[:, 0:5],
                        )
                    )
                    
                    

                # For multiclass, need to re-encode Y_mb as a one-hot matrix of 32x5
                # Return Tilde as 32x10 and use Tilde[:, 5:] for Tilde[:, 1] and Tilde[:, 0:5] for Tilde[:, 0]

                G_loss = G_loss_R + self.alpha * G_loss_GAN

                if self.counterfactual_invariance_generator:
                    unique_clusters = list(set(cluster_mb))
                    mmd_loss = 0
                    for comb in combinations(unique_clusters, 2):
                        cluster1 = comb[0]
                        cluster2 = comb[1]
                        Tilde_cluster1 = Tilde[cluster_mb == cluster1, :]
                        Tilde_cluster2 = Tilde[cluster_mb == cluster2, :]
                        mmd_loss += compute_mmd(Tilde_cluster1, Tilde_cluster2)
                    
                    G_loss += penalty_multiplier * mmd_loss

                    # if penalty_multiplier > 1.0:
                    #     G_loss /= penalty_multiplier

                if it % 100 == 0:
                    log.debug(f"Generator loss epoch {it}: {D_loss} {G_loss}")
                    if torch.isnan(D_loss).any():
                        raise RuntimeError("counterfactual_discriminator generated NaNs")

                    if torch.isnan(G_loss).any():
                        raise RuntimeError("counterfactual_generator generated NaNs")

                G_loss.backward()

            self.DG_solver.step()
            if self.lm_gen:
                if X_text is not None:
                    self.DG_lm_solver.step()
                if X_image is not None:
                    self.DG_vm_solver.step()

        # Train I and ID
        if self.IRM:
            dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE)

        print ('Training I')
        if not self.lm_gen and (X_text is not None or X_image is not None):
            Train_X = Train_X[:, 0:120]

        for it in tqdm(range(self.num_iterations)):
            self.I_solver.zero_grad()
            if X_text is not None:
                self.I_lm_solver.zero_grad()
            if X_image is not None:
                self.I_vm_solver.zero_grad()
            # if self.IRM:
            penalty_multiplier = 2.0 * it / self.num_iterations ** 1.6
            # elif self.counterfactual_invariance:
                # penalty_multiplier = 2.0 * it / self.num_iterations ** 3
            # penalty_multiplier = 2.0 * it / self.num_iterations ** 2.0
            # next - try increasing the penalty? to like 8.0 * or something?
            if self.IRM:
                error = 0
                penalty = 0

            if X_text is None and X_image is None:
                X_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=False, dim_output=dim_output)
                X_text_mb = None
                X_image_mb = None
            elif X_text is not None and X_image is not None:
                X_mb, X_text_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
            elif X_text is not None and X_image is None:
                X_mb, X_text_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                X_image_mb = None
            elif X_text is None and X_image is not None:
                X_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                X_text_mb = None
            
            X_mb = self._process_X_text(X_mb, X_text_mb)
            X_mb = self._process_X_image(X_mb, X_image_mb)

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)
            hat = self.inference_nets(X_mb)

            I_loss1: torch.Tensor = 0
            I_loss2: torch.Tensor = 0

            if self.IRM:
                unique_clusters = list(set(cluster_mb))
                for cluster in unique_clusters:
                    dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE).requires_grad_()
                    T_cluster = T_mb[cluster_mb == cluster, :]
                    Y_cluster = Y_mb[cluster_mb == cluster, :]
                    Tilde_cluster = Tilde[cluster_mb == cluster, :]
                    hat_cluster = hat[cluster_mb == cluster, :]
                    if task == 'classification' or task == 'regression':
                        I_loss1 = nn.MSELoss()(
                                T_cluster * Y_cluster + (1 - T_cluster) * Tilde_cluster[:, 1].reshape([-1, 1]),
                                (hat_cluster[:, 1] * dummy_w).reshape([-1, 1]),
                            )
                        I_loss2 = nn.MSELoss()(
                                (1 - T_cluster) * Y_cluster + T_cluster * Tilde_cluster[:, 0].reshape([-1, 1]),
                                (hat_cluster[:, 0] * dummy_w).reshape([-1, 1]),
                            )
                    elif task == 'multiclass':
                        I_loss1 = nn.MSELoss()(
                                T_cluster * Y_cluster + (1 - T_cluster) * Tilde_cluster[:, 5:],
                                (hat_cluster[:, 5:] * dummy_w),
                            )
                        I_loss2 = nn.MSELoss()(
                                (1 - T_cluster) * Y_cluster + T_cluster * Tilde_cluster[:, 0:5],
                                (hat_cluster[:, 0:5] * dummy_w),
                            )
                    I_loss = I_loss1 + self.beta * I_loss2
                    penalty += compute_irm_penalty(I_loss, dummy_w)
                    error += I_loss.mean()

                if it % 100 == 0:
                    log.debug(f"Inference loss epoch {it}: {I_loss}")

                error += penalty_multiplier * penalty
                if penalty_multiplier > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    error /= penalty_multiplier

                error.backward()
            
            else:
                if task == 'classification' or task == 'regression':
                    I_loss1 = torch.mean(
                        nn.MSELoss()(
                            T_mb * Y_mb + (1 - T_mb) * Tilde[:, 1].reshape([-1, 1]),
                            hat[:, 1].reshape([-1, 1]),
                        )
                    )
                    I_loss2 = torch.mean(
                        nn.MSELoss()(
                            (1 - T_mb) * Y_mb + T_mb * Tilde[:, 0].reshape([-1, 1]),
                            hat[:, 0].reshape([-1, 1]),
                        )
                    )
                elif task == 'multiclass':
                    I_loss1 = torch.mean(
                        nn.MSELoss()(
                            T_mb * Y_mb + (1 - T_mb) * Tilde[:, 5:],
                            hat[:, 5:],
                        )
                    )
                    I_loss2 = torch.mean(
                        nn.MSELoss()(
                            (1 - T_mb) * Y_mb + T_mb * Tilde[:, 0:5],
                            hat[:, 0:5],
                        )
                    )
                    
                I_loss = I_loss1 + self.beta * I_loss2

                if self.counterfactual_invariance:
                    unique_clusters = list(set(cluster_mb))
                    mmd_loss = 0
                    for comb in combinations(unique_clusters, 2):
                        cluster1 = comb[0]
                        cluster2 = comb[1]
                        hat_cluster1 = hat[cluster_mb == cluster1, :]
                        hat_cluster2 = hat[cluster_mb == cluster2, :]
                        mmd_loss += compute_mmd(hat_cluster1, hat_cluster2)
                    
                    I_loss += penalty_multiplier * mmd_loss
                    # I_loss += mmd_loss

                    # if penalty_multiplier > 1.0:
                    #     I_loss /= penalty_multiplier

                if it % 100 == 0:
                    log.debug(f"Inference loss epoch {it}: {I_loss}")

                I_loss.backward()

            self.I_solver.step()
            if X_text is not None:
                self.I_lm_solver.step()
            if X_image is not None:
                self.I_vm_solver.step()

        return self

    def forward(self, X: torch.Tensor, X_text, X_image, batch_size) -> torch.Tensor:
        y_hat_list = []
        for idx in tqdm(range(0, X.shape[0], batch_size)):
            X_mb = X[idx:(idx+batch_size), :]
            if X_text is not None:
                X_text_mb = X_text[idx:(idx+batch_size)]
                X_text_mb = self.tokenizer(X_text_mb.tolist(), padding='max_length', truncation=True, return_tensors='pt')
            else:
                X_text_mb = None
            if X_image is not None:
                X_image_mb = X_image[idx:(idx+batch_size)]
                img_list = [Image.open(os.path.join(self.data_dir, 'airbnb/images/{}'.format(filepath.split('/')[-1]))).convert(mode='RGB')
                    for filepath in X_image_mb]
                X_image_mb = self.feature_extractor(images=img_list, return_tensors='pt')
                for img in img_list:
                    img.close()
                    del(img)
            else:
                X_image_mb = None
            with torch.no_grad():
                X_mb = self._process_X_text(X_mb, X_text_mb, eval=True)
                X_mb = self._process_X_image(X_mb, X_image_mb, eval=True)
                X_mb = self._check_tensor(X_mb).float()
                y_hat_mb = self.inference_nets(X_mb).detach()
            y_hat_list.append(y_hat_mb)
        
        y_hat = torch.cat(y_hat_list)
            
        # with torch.no_grad():
        #     X = self._check_tensor(X).float()
        #     y_hat = self.inference_nets(X).detach()
            
        return y_hat

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).to(DEVICE)

class SelfTrainingCounterfactualGAN(CounterfactualGAN):
    def _fit(self,
        X: torch.Tensor,
        Treatment: torch.Tensor,
        Y: torch.Tensor,
        clusters: torch.Tensor,
        dim_output: int,
        task,
        X_text: np.ndarray = None,
        X_image: np.ndarray = None,
    ):
        Train_X = self._check_tensor(X).float()
        Train_T = self._check_tensor(Treatment).float().reshape([-1, 1])
        Train_Y = self._check_tensor(Y).float().reshape([-1, 1])

        # Encode
        min_t_val = Train_T.min()
        Train_T = (Train_T > min_t_val).float()

        # Iterations
        # Train G and D first
        self.counterfactual_generator.train()
        self.counterfactual_discriminator.train()
        self.inference_nets.train()
        if X_text is not None:
            self.language_model.train()
        if X_image is not None:
            self.image_model.train()

        # if self.IRM_generator:
        #     dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE)
        for selftrain_it in tqdm(range(self.num_selftrain_iterations)):
            print ('Training G and D')
            if selftrain_it > 0:
                if self.share_weights:
                    teacher_weights = self.counterfactual_generator.state_dict()
                    student_weights = self.inference_nets.state_dict()
                    for key, value in student_weights.items():
                        if teacher_weights[key].shape != student_weights[key].shape:
                            teacher_weights[key][0:student_weights[key].shape[0], 0:student_weights[key].shape[1]] = student_weights[key]
                        else:
                            teacher_weights[key] = student_weights[key]
                    self.counterfactual_generator.load_state_dict(teacher_weights)
                    
            self.counterfactual_generator.train()
            self.counterfactual_discriminator.train()
            for it in tqdm(range(self.num_iterations)):
                self.DG_solver.zero_grad()
                if self.lm_gen:
                    if X_text is not None:
                        self.DG_lm_solver.zero_grad()
                    if X_image is not None:
                        self.DG_vm_solver.zero_grad()

                # penalty_multiplier = 2.0 * it / self.num_iterations ** 2.0
                # if self.IRM_generator:
                penalty_multiplier = 2.0 * it / self.num_iterations ** 1.6
                # elif self.counterfactual_invariance_generator:
                #     penalty_multiplier = 2.0 * it / self.num_iterations ** 3

                if self.IRM_generator:                
                    error = 0
                    penalty = 0
                if not self.lm_gen or (X_text is None and X_image is None):
                    X_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=False, dim_output=dim_output)
                    X_text_mb = None
                    X_image_mb = None
                elif X_text is not None and X_image is not None:
                    X_mb, X_text_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                elif X_text is not None and X_image is None:
                    X_mb, X_text_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                    X_image_mb = None
                elif X_text is None and X_image is not None:
                    X_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                    X_text_mb = None
                
                for kk in range(self.num_discr_iterations):
                    if self.lm_gen:
                        X_mb_concat = self._process_X_text(X_mb, X_text_mb)
                        X_mb_concat = self._process_X_image(X_mb_concat, X_image_mb)
                    else:
                        X_mb_concat = X_mb

                    self.DG_solver.zero_grad()
                    if self.lm_gen:
                        if X_text is not None:
                            self.DG_lm_solver.zero_grad()
                        if X_image is not None:
                            self.DG_vm_solver.zero_grad()

                    Tilde = self.counterfactual_generator(X_mb_concat, T_mb, Y_mb).clone()
                    D_out = self.counterfactual_discriminator(X_mb_concat, T_mb, Y_mb, Tilde)

                    if torch.isnan(Tilde).any():
                        raise RuntimeError("counterfactual_generator generated NaNs")
                    if torch.isnan(D_out).any():
                        raise RuntimeError("counterfactual_discriminator generated NaNs")

                    D_loss = nn.BCELoss()(D_out, T_mb)
                    D_loss.backward()

                    self.DG_solver.step()
                    if self.lm_gen:
                        if X_text is not None:
                            self.DG_lm_solver.step()
                        if X_image is not None:
                            self.DG_vm_solver.step()

                if self.lm_gen:
                    X_mb_concat = self._process_X_text(X_mb, X_text_mb)
                    X_mb_concat = self._process_X_image(X_mb_concat, X_image_mb)
                else:
                    X_mb_concat = X_mb
                Tilde = self.counterfactual_generator(X_mb_concat, T_mb, Y_mb)
                D_out = self.counterfactual_discriminator(X_mb_concat, T_mb, Y_mb, Tilde)
                D_loss = nn.BCELoss()(D_out, T_mb)

                G_loss_GAN = D_loss
                
                if selftrain_it > 0:
                    # self.inference_nets.eval()
                    with torch.no_grad():
                        y_learned_mb = self.inference_nets(X_mb)
                        y_combined_mb = torch.zeros(y_learned_mb.shape).to(DEVICE)
                        y_combined_mb[np.arange(len(y_learned_mb)), torch.flatten(T_mb).to(int)] = torch.flatten(Y_mb).float()
                        y_combined_mb[np.arange(len(y_learned_mb)), torch.flatten(torch.abs(1 - T_mb)).to(int)] = y_learned_mb[np.arange(len(y_learned_mb)), torch.flatten(torch.abs(1 - T_mb)).to(int)]

                if self.IRM_generator:
                    unique_clusters = list(set(cluster_mb))
                    for cluster in unique_clusters:
                        dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE).requires_grad_()
                        T_cluster = T_mb[cluster_mb == cluster, :]
                        Y_cluster = Y_mb[cluster_mb == cluster, :]
                        Tilde_cluster = Tilde[cluster_mb == cluster, :]
                        
                        if selftrain_it > 0:
                            y_combined_cluster = y_combined_mb[cluster_mb == cluster, :]
                            G_loss_R = nn.MSELoss()(y_combined_cluster[:, 1], Tilde_cluster[:, 1] * dummy_w) + nn.MSELoss()(y_combined_cluster[:, 0], Tilde_cluster[:, 0] * dummy_w)
                        else:
                            G_loss_R = nn.MSELoss()(
                                    Y_cluster,
                                    (T_cluster * Tilde_cluster[:, 1].reshape([-1, 1])
                                    + (1.0 - T_cluster) * Tilde_cluster[:, 0].reshape([-1, 1])) * dummy_w,
                                )

                        G_loss = G_loss_R + self.alpha * G_loss_GAN
                        penalty += compute_irm_penalty(G_loss, dummy_w)
                        error += G_loss.mean()

                    if it % 100 == 0:
                        log.debug(f"Generator loss epoch {it}: {D_loss} {G_loss}")
                        if torch.isnan(D_loss).any():
                            raise RuntimeError("counterfactual_discriminator generated NaNs")

                        if torch.isnan(G_loss).any():
                            raise RuntimeError("counterfactual_generator generated NaNs")
                    
                    if penalty_multiplier > 1.0:
                        error /= penalty_multiplier

                    (error + penalty_multiplier * penalty).backward()

                else:
                    if selftrain_it > 0:
                        G_loss_R = torch.mean(
                            nn.MSELoss()(y_combined_mb[:, 1], Tilde[:, 1]) + nn.MSELoss()(y_combined_mb[:, 0], Tilde[:, 0])
                        )
                    else:
                        G_loss_R = torch.mean(
                            nn.MSELoss()(
                                Y_mb,
                                T_mb * Tilde[:, 1].reshape([-1, 1])
                                + (1.0 - T_mb) * Tilde[:, 0].reshape([-1, 1]),
                            )
                        )

                    G_loss = G_loss_R + self.alpha * G_loss_GAN

                    if self.counterfactual_invariance_generator:
                        unique_clusters = list(set(cluster_mb))
                        mmd_loss = 0
                        for comb in combinations(unique_clusters, 2):
                            cluster1 = comb[0]
                            cluster2 = comb[1]
                            Tilde_cluster1 = Tilde[cluster_mb == cluster1, :]
                            Tilde_cluster2 = Tilde[cluster_mb == cluster2, :]
                            mmd_loss += compute_mmd(Tilde_cluster1, Tilde_cluster2)
                        
                        G_loss += penalty_multiplier * mmd_loss

                        # if penalty_multiplier > 1.0:
                        #     G_loss /= penalty_multiplier

                    if it % 100 == 0:
                        log.debug(f"Generator loss epoch {it}: {D_loss} {G_loss}")
                        if torch.isnan(D_loss).any():
                            raise RuntimeError("counterfactual_discriminator generated NaNs")

                        if torch.isnan(G_loss).any():
                            raise RuntimeError("counterfactual_generator generated NaNs")

                    G_loss.backward()

                self.DG_solver.step()
                if self.lm_gen:
                    if X_text is not None:
                        self.DG_lm_solver.step()
                    if X_image is not None:
                        self.DG_vm_solver.step()

            # Train I and ID
            if self.share_weights:
                teacher_weights = self.counterfactual_generator.state_dict()
                student_weights = self.inference_nets.state_dict()
                for key, value in teacher_weights.items():
                    if teacher_weights[key].shape != student_weights[key].shape:
                        teacher_weights[key] = teacher_weights[key][0:student_weights[key].shape[0], 0:student_weights[key].shape[1]]
                self.inference_nets.load_state_dict(teacher_weights)
                
            self.inference_nets.train()
            # self.counterfactual_generator.eval()
            # self.counterfactual_discriminator.eval()
            if self.IRM:
                dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE)

            print ('Training I')
            if not self.lm_gen and (X_text is not None or X_image is not None):
                Train_X = Train_X[:, 0:120]

            for it in tqdm(range(self.num_iterations)):
                self.I_solver.zero_grad()
                if X_text is not None:
                    self.I_lm_solver.zero_grad()
                if X_image is not None:
                    self.I_vm_solver.zero_grad()
                # if self.IRM:
                penalty_multiplier = 2.0 * it / self.num_iterations ** 1.6
                # elif self.counterfactual_invariance:
                    # penalty_multiplier = 2.0 * it / self.num_iterations ** 3
                # penalty_multiplier = 2.0 * it / self.num_iterations ** 2.0
                # next - try increasing the penalty? to like 8.0 * or something?
                if self.IRM:
                    error = 0
                    penalty = 0

                if X_text is None and X_image is None:
                    X_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=False, dim_output=dim_output)
                    X_text_mb = None
                    X_image_mb = None
                elif X_text is not None and X_image is not None:
                    X_mb, X_text_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                elif X_text is not None and X_image is None:
                    X_mb, X_text_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                    X_image_mb = None
                elif X_text is None and X_image is not None:
                    X_mb, X_image_mb, T_mb, Y_mb, cluster_mb = self._sample_minibatch(Train_X, X_text, X_image, Train_T, Train_Y, clusters, gen_embeds=True, dim_output=dim_output)
                    X_text_mb = None
                
                X_mb = self._process_X_text(X_mb, X_text_mb)
                X_mb = self._process_X_image(X_mb, X_image_mb)
                
                Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)
                hat = self.inference_nets(X_mb)

                I_loss1: torch.Tensor = 0
                I_loss2: torch.Tensor = 0

                if self.IRM:
                    unique_clusters = list(set(cluster_mb))
                    for cluster in unique_clusters:
                        dummy_w = nn.Parameter(torch.Tensor([1.0])).to(DEVICE).requires_grad_()
                        T_cluster = T_mb[cluster_mb == cluster, :]
                        Y_cluster = Y_mb[cluster_mb == cluster, :]
                        Tilde_cluster = Tilde[cluster_mb == cluster, :]
                        hat_cluster = hat[cluster_mb == cluster, :]

                        I_loss1 = nn.MSELoss()(
                                T_cluster * Y_cluster + (1 - T_cluster) * Tilde_cluster[:, 1].reshape([-1, 1]),
                                (hat_cluster[:, 1] * dummy_w).reshape([-1, 1]),
                            )
                        I_loss2 = nn.MSELoss()(
                                (1 - T_cluster) * Y_cluster + T_cluster * Tilde_cluster[:, 0].reshape([-1, 1]),
                                (hat_cluster[:, 0] * dummy_w).reshape([-1, 1]),
                            )
                        I_loss = I_loss1 + self.beta * I_loss2
                        penalty += compute_irm_penalty(I_loss, dummy_w)
                        error += I_loss.mean()

                    if it % 100 == 0:
                        log.debug(f"Inference loss epoch {it}: {I_loss}")

                    error += penalty_multiplier * penalty
                    if penalty_multiplier > 1.0:
                        # Rescale the entire loss to keep gradients in a reasonable range
                        error /= penalty_multiplier

                    error.backward()
                
                else:

                    I_loss1 = torch.mean(
                        nn.MSELoss()(
                            T_mb * Y_mb + (1 - T_mb) * Tilde[:, 1].reshape([-1, 1]),
                            hat[:, 1].reshape([-1, 1]),
                        )
                    )
                    I_loss2 = torch.mean(
                        nn.MSELoss()(
                            (1 - T_mb) * Y_mb + T_mb * Tilde[:, 0].reshape([-1, 1]),
                            hat[:, 0].reshape([-1, 1]),
                        )
                    )
                    
                    I_loss = I_loss1 + self.beta * I_loss2

                    if self.counterfactual_invariance:
                        unique_clusters = list(set(cluster_mb))
                        mmd_loss = 0
                        for comb in combinations(unique_clusters, 2):
                            cluster1 = comb[0]
                            cluster2 = comb[1]
                            hat_cluster1 = hat[cluster_mb == cluster1, :]
                            hat_cluster2 = hat[cluster_mb == cluster2, :]
                            mmd_loss += compute_mmd(hat_cluster1, hat_cluster2)
                        
                        I_loss += penalty_multiplier * mmd_loss
                        # I_loss += mmd_loss

                        # if penalty_multiplier > 1.0:
                        #     I_loss /= penalty_multiplier

                    if it % 100 == 0:
                        log.debug(f"Inference loss epoch {it}: {I_loss}")

                    I_loss.backward()

                self.I_solver.step()
                if X_text is not None:
                    self.I_lm_solver.step()
                if X_image is not None:
                    self.I_vm_solver.step()

        return self

class CounterfactualGANText(CounterfactualGAN):
    def _process_X_text(self, X_mb, X_text_mb, eval=False):
        if eval:
            self.language_model.eval()
        else:
            self.language_model.train()

        X_text_mb.to(DEVICE)
        X_text_mb = self.language_model(**X_text_mb).last_hidden_state.mean(axis=1)
        X_mb = torch.cat((X_mb, X_text_mb), axis=1)
        
        return X_mb

    def _process_X_image(self, X_mb, X_image_mb, eval=False):
        return X_mb

class CounterfactualGANMultimodal(CounterfactualGAN):
    def _process_X_text(self, X_mb, X_text_mb, eval=False): 
        if eval:
            self.language_model.eval()
        else:
            self.language_model.train()

        X_text_mb.to(DEVICE)
        X_text_mb = self.language_model(**X_text_mb).last_hidden_state.mean(axis=1)
        X_mb = torch.cat((X_mb, X_text_mb), axis=1)
        
        return X_mb

    def _process_X_image(self, X_mb, X_image_mb, eval=False):
        if eval:
            self.image_model.eval()
        else:
            self.image_model.train()
        
        X_image_mb.to(DEVICE)
        X_image_mb = self.image_model(**X_image_mb).last_hidden_state.mean(axis=1)
        X_mb = torch.cat((X_mb, X_image_mb), axis=1)

        return X_mb

'''
Potentially use this to replace DG and I_solver if we need different learning rates for the language model

class MultipleOptimizer(object):
    def __init__(*op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


opt = MultipleOptimizer(optimizer1(params1, lr=lr1), 
                        optimizer2(params2, lr=lr2))
'''