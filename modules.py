from collections import defaultdict
# document loaders
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextLoaderWithMetadata(TextLoader):
    def __init__(self, file_path: str, metadata_func=None, **kwargs):
        super().__init__(file_path=file_path, **kwargs)
        self.metadata_func = metadata_func
    
    def load(self) -> list[Document]:
        # load the documents as TextLoader
        docs = super.load()
        # if a metadata function is specified
        if self.metadata_func is not None:
            for d in docs:
                d.metadata = self.metadata_func(d, d.metadata)
        
        return docs
    
    def lazy_load(self):
        # for each document loaded using TextLoader
        for doc in super().lazy_load():
            # if a metadata function is specified
            if self.metadata_func is not None:
                # apply the metadata function to the document
                doc.metadata = self.metadata_func(doc, dict(doc.metadata))
            yield doc


class TextSplitterWithLineNum(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def split_documents(self, documents):
        split = super().split_documents(documents)

        # create a mapping for every document
        line_num_map = {}
        for d in documents:
            file_name = d.metadata['source']
            m = defaultdict(list)
            # iterate through each line and give it a mapping
            for idx, line in enumerate(d.page_content.split('\n')):
                m[line.strip()].append(idx + 1)
            line_num_map[file_name] = m

        # for each chunk find the line index mapping for the first and last line
        for chunk in split:
            source = chunk.metadata['source']
            lines = chunk.page_content.strip().split('\n')
            # get the first and last line of the chunk
            first, last = line_num_map[source][lines[0].strip()], line_num_map[source][lines[-1].strip()]

            # find the start index and 
            start = first.pop(0)
            # choose the first matching end line at or after start
            end = next(i for i in last if i >= start)

            chunk.metadata['source'] = {"file": source, 'lines': [start, end]}
        return split