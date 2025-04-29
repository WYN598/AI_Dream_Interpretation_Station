# import sys
# print(sys.path)
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
# pip install llama-index-indices-managed-llama-cloud

index = LlamaCloudIndex(
  name="Dream-Rag",
  project_name="Default",
  organization_id="0d8d2dff-9055-45db-b02c-a3b573bfde48",
  api_key="llx-a7jTrhbFK1QPDdbfNfsVbEI3soTvvmN7GajiRVbIsCWZXcbY",
)
query = "What is Don Juan's idea?"
nodes = index.as_retriever().retrieve(query)
response = index.as_query_engine().query(query)

print("Nodes:")
for node in nodes:
    print(node)
print("Response:")
print(response)
