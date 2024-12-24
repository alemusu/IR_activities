# PyTerrier init
import pyterrier as pt

if not pt.started():
    pt.init()


# Vaswani dataset download + limit queries to 50 topics
dataset = pt.get_dataset("vaswani")
topics = dataset.get_topics().head(50)
qrels = dataset.get_qrels()


# index Vaswani collection with colBERT
import pyterrier_colbert.indexing

colbert_indexer = pyterrier_colbert.indexing.ColBERTIndexer(checkpoint=checkpoint,
                                                            index_root="/content",
                                                            index_name="colbert_index",
                                                            chunksize=3)
colbert_indexer.index(dataset.get_corpus_iter())


# create in a BM25 baseline transformer, and the ColBERT retrieve transformer
from pyterrier_colbert.ranking import ColBERTFactory

bm25_terrier_stemmed = pt.BatchRetrieve.from_dataset('vaswani',
                                                     'terrier_steemed_text',
                                                     wmodel='BM25',
                                                     metadata=['docno','text'])
factory = ColBERTFactory.from_dataset('vaswani', 'colbert_uog44k')
colbert_e2e = factory.end_to_end()


# retrieve top 10 ranked docs & compute effectiveness metrics
pt.Experiment(
    [bm25_terrier_stemmed % 10, colbert_e2e % 10],
    topics,
    qrels,
    eval_metrics=["map", "recip_rank", "p_10", "ndcg_cut_10" "mrt"],
    names=['BM25', 'ColBERT']
)


# visualizing ColBERT
query = 'What is the origin of covid 19'
document = 'Origin of the COVID-19 virus has been intensely debated in the scientific community since the first infected cases were detected in December 2019. The disease has caused a global pandemic, leading to deaths of thousands of people across the world and thus finding origin of this novel coronavirus is'
figure = factory.explain_text(query,document)