import torch
import cv2
import numpy as np

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fh = target_layer.register_forward_hook(self.forward_hook)
        self.bh = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):

        self.model.zero_grad()

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        prob = probs[0, pred].item()

        score = logits[0, pred]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=(2,3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1)

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()[0]

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam, pred, prob