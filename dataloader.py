import lmdb
import multiprocessing as mp
import time
import random
import numpy as np


class Dataloader:
    def __init__(self, 
                 lmdb_path, 
                 batch_size,
                 num_workers,
                 src_tokenizer,
                 en_tokenizer,
                 shuffle=True,
                 augment=False
                ):
        self.lmdb_path = lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.augment = augment
        self.len = None
        self.processes = []
        
    def __len__(self):
        if self.len is None:
            env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                self.len = int(txn.get(f"len".encode()).decode())
        return self.len

    def get_src_lengths(self):
        lengths = []
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            for i in range(len(self)):
                src = txn.get(f"{i}_src".encode())
                lengths.append((i, len(src)))
        return lengths

    @staticmethod
    def worker_function(db_path, src_tokenizer, en_tokenizer, request_queue, result_queue):
        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        while True:
            request = request_queue.get()
            if request is None:
                break
            
            sequence_number = request['sequence_number']
            indices = request['indices']
            
            data_batch = []
            with env.begin(write=False) as txn:
                for index in indices:
                    src = f"{index}_src".encode()
                    en = f"{index}_en".encode()
                    src = txn.get(src)
                    en = txn.get(en)
                    
                    data_batch.append((src.decode(), en.decode()))                
            
            result_queue.put({'sequence_number': sequence_number, 'data': data_batch})
        
        env.close()
    
    def __iter__(self):
                   
        buffer = {}
        sequence_number = 0
        next_sequence_number = 0

        request_queue = mp.Queue()
        result_queue = mp.Queue()
        
        for i in range(self.num_workers):
            p = mp.Process(target=Dataloader.worker_function, args=(self.lmdb_path, request_queue, result_queue))
            self.processes.append(p)
            p.start()
        

        if self.shuffle:
            # for efficient padding, batch by similar length
            chunk_len = 1000
            lengths = self.get_src_lengths()
            lengths.sort(key=lambda x: x[1])
            indices = np.array([x[0] for x in lengths])
            for i in range(0, len(self), chunk_len):
                np.random.shuffle(lengths[i:i+chunk_len])
            indices = indices.tolist()
            batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            random.shuffle(batches)
        else:
            indices = list(range(len(self)))
            batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]

        # fill the queue with initial batches
        for batch_indices in batches:
            request_queue.put({'sequence_number': sequence_number, 'indices': batch_indices})
            sequence_number += 1
    
        # ensure batches are processed in order for reproducibility
        while next_sequence_number < sequence_number:
            while next_sequence_number not in buffer:
                result = result_queue.get()
                buffer[result['sequence_number']] = result['data']
            
            data_batch = buffer.pop(next_sequence_number)
            next_sequence_number += 1
            yield data_batch
    
        # Signal workers to stop
        for _ in range(self.num_workers):
            request_queue.put(None)
    
        for p in self.processes:
            p.join()
        self.processes = []

    def __del__(self):
        for p in self.processes:
            p.kill()