import torch
import torch.nn as nn
from transformers import SwinForImageClassification, SwinConfig
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

class SwinLoRAModel:
    """
    Swin Transformer with LoRA for image classification tasks.
    Supports various Swin Transformer architectures.
    """

    def __init__(self, model_name="microsoft/swin-base-patch4-window7-224",
                 num_labels=10, lora_r=8, lora_alpha=16, lora_dropout=0.1,
                 target_modules=None, use_lora=True):
        """
        Initialize Swin Transformer model with optional LoRA adaptation.

        Args:
            model_name: Pretrained Swin model name
            num_labels: Number of output classes
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA
            use_lora: Whether to apply LoRA adaptation
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_lora = use_lora

        # Default target modules for Swin Transformer
        if target_modules is None:
            target_modules = ['attention.self.query', 'attention.self.value']

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["classifier"],
        )

    def get_model(self):
        """
        Get Swin Transformer model with optional LoRA adaptation.

        Returns:
            Configured Swin Transformer model
        """
        try:
            # Load pretrained Swin model
            model = SwinForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                ignore_mismatched_sizes=True
            )

            # Replace classifier if needed
            if hasattr(model, 'classifier'):
                # Swin models typically have classifier as nn.Linear
                if isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, self.num_labels)
                else:
                    # Handle different classifier types
                    in_features = model.config.hidden_size
                    model.classifier = nn.Linear(in_features, self.num_labels)

            # Apply LoRA if requested
            if self.use_lora:
                model = get_peft_model(model, self.lora_config)
                logger.info(f"Applied LoRA to Swin model: {self.model_name}")
            else:
                logger.info(f"Loaded standard Swin model: {self.model_name}")

            return model

        except Exception as e:
            logger.error(f"Error loading Swin model {self.model_name}: {str(e)}")
            raise

    def get_processor(self):
        """
        Get image processor for Swin Transformer model.

        Returns:
            Swin image processor
        """
        from transformers import AutoImageProcessor

        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            return processor
        except Exception as e:
            logger.error(f"Error loading Swin processor: {str(e)}")
            raise

    def count_trainable_parameters(self, model):
        """
        Count trainable parameters in the model.

        Args:
            model: The model to count parameters for

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
