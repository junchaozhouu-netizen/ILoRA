import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

class RoBERTaLoRAModel:
    """
    RoBERTa with LoRA for sequence classification tasks.
    Supports various RoBERTa architectures for NLP tasks.
    """
    
    def __init__(self, model_name="roberta-base", num_labels=2, 
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, 
                 target_modules=None, use_lora=True, max_length=512):
        """
        Initialize RoBERTa model with optional LoRA adaptation.
        
        Args:
            model_name: Pretrained RoBERTa model name
            num_labels: Number of output classes
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA
            use_lora: Whether to apply LoRA adaptation
            max_length: Maximum sequence length for tokenizer
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_lora = use_lora
        self.max_length = max_length
        
        # Default target modules for RoBERTa
        if target_modules is None:
            target_modules = ['query', 'value']
            
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["classifier", "score"],
        )
        
    def get_model(self):
        """
        Get RoBERTa model with optional LoRA adaptation.
        
        Returns:
            Configured RoBERTa model
        """
        try:
            # Load pretrained RoBERTa model
            model = RobertaForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            
            # Apply LoRA if requested
            if self.use_lora:
                model = get_peft_model(model, self.lora_config)
                logger.info(f"Applied LoRA to RoBERTa model: {self.model_name}")
            else:
                logger.info(f"Loaded standard RoBERTa model: {self.model_name}")
                
            return model
            
        except Exception as e:
            logger.error(f"Error loading RoBERTa model {self.model_name}: {str(e)}")
            raise
    
    def get_tokenizer(self):
        """
        Get tokenizer for RoBERTa model.
        
        Returns:
            RoBERTa tokenizer
        """
        try:
            tokenizer = RobertaTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length
            )
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading RoBERTa tokenizer: {str(e)}")
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

    def prepare_inputs(self, texts, tokenizer, device='cuda'):
        """
        Prepare inputs for RoBERTa model.
        
        Args:
            texts: List of input texts
            tokenizer: RoBERTa tokenizer
            device: Target device
            
        Returns:
            Dictionary of model inputs
        """
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {k: v.to(device) for k, v in inputs.items()}