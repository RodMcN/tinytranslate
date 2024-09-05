from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import lmdb

def get_tokenizers(lmdb_path, src_vocab_size=120_000, en_vocab_size=30_000):
    src_tok = train_tokenizer(lmdb_path, "src", src_vocab_size)
    en_tok = train_tokenizer(lmdb_path, "en", en_vocab_size)
    return src_tok, en_tok


def train_tokenizer(lmdb_path, lang, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<sos>", "<eos>"])
    
    tokenizer.train_from_iterator(read_dataset(lmdb_path, lang), trainer=trainer)

    if lang != "src":
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<sos> $A <eos>",
            special_tokens=[("<sos>", tokenizer.token_to_id("<sos>")),
                           ("<eos>", tokenizer.token_to_id("<eos>"))] 
        )

    
    return tokenizer
    
def read_dataset(lmdb_path, lang):
    if lang not in ("en", "src"):
        raise ValueError("'lang' must be either 'en' or 'src'")
        
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        total = int(txn.get(f"len".encode()).decode())
        
        for i in range(total):
            key = f"{i}_{lang}".encode()
            val = txn.get(key).decode()
            yield val
