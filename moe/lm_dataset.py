import json
import os
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        samples = []
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # 验证数据格式
                    if isinstance(data, dict) and 'text' in data:
                        samples.append(data)
                    else:
                        print(f"Warning: Line {line_num} does not contain 'text' field, skipping")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    print(f"Problematic line: {line[:100]}...")  # 显示前100个字符
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error at line {line_num}: {e}")
                    continue
        
        if not samples:
            raise ValueError(f"No valid samples found in {data_path}")
        
        print(f"Loaded {len(samples)} samples from {data_path}")
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 确保 text 字段存在且不为空
        text = sample.get('text', '')
        if not text:
            text = "Empty text"  # 提供默认值
        
        encoding = self.tokenizer(
            str(text),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)

        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.float)
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length    
        self.samples = self.load_data(jsonl_path)

        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'conversations' in data:
                        samples.append(data)
                    else:
                        print(f"Warning: Line {line_num} does not contain 'conversations' field, skipping")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error at line {line_num}: {e}")
                    continue
        
        if not samples:
            raise ValueError(f"No valid samples found in {path}")
            
        print(f"Loaded {len(samples)} samples from {path}")
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn['content']})
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False)
    
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0

        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i = len(self.bos_id)
                end = start
                while i < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)

            else:
                i += 1
        
        return loss_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)


        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)

        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

    
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and 'chosen' in obj and 'rejected' in obj:
                        self.data.append(obj)
                    else:
                        print(f"Warning: Line {line_num} missing 'chosen' or 'rejected' field, skipping")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error at line {line_num}: {e}")
                    continue
        
        if not self.data:
            raise ValueError(f"No valid samples found in {file_path}")
            
        print(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']

        rejected = item['rejected']
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        loss_mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        loss_mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "loss_mask_chosen": loss_mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "loss_mask_rejected": loss_mask_rejected
        }
    
    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0

        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i = len(self.bos_id)
                end = start
                while i < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)

            else:
                i += 1
        
        return loss_mask
    
    
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids  # 修复缺失的 _ids
    
    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = [] 
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'conversations' in data:
                        samples.append(data)
                    else:
                        print(f"Warning: Line {line_num} does not contain 'conversations' field, skipping")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Unexpected error at line {line_num}: {e}")
                    continue
        
        if not samples:
            raise ValueError(f"No valid samples found in {path}")
            
        print(f"Loaded {len(samples)} samples from {path}")
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']

        return self.tokenizer.apply_chat_template(
            messages[:-1], 
            tokenize=False,
            add_generation_prompt=True,
            ), answer

    
    def __getitem__(self, index):
        sample = self.samples[index]

        prompt, answer = self._create_chat_prompt(sample["conversations"])

        return {
            "prompt": prompt,
            "answer": answer
        }
    

if __name__ == "__main__":
    pass


