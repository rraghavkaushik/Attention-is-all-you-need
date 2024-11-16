# Attention-is-all-you-need : Tensorflow Implementation
My implementation of the Transformer architecture based on the paper "Attention Is All You Need"

## Key Points
- The idea behind multi-head attention is to allow the attention function to extract information from different representation subspaces, which would otherwise be impossible with a single attention head.
- Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
  <img width="591" alt="image" src="https://github.com/user-attachments/assets/e08bb4c1-9945-4916-923c-55f839479a1a">

- An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key (As described in the paper).


### Scaled Dot Product Attention
<img width="377" alt="image" src="https://github.com/user-attachments/assets/2382c017-44ca-4c5b-9f05-223af768d278">

### MultiHead Attention
<img width="377" alt="image" src="https://github.com/user-attachments/assets/2f7dbeaf-0eba-40a9-bb96-6fff7246ab99">

## Positional Encoding

for more information refer to the file 'positional_encoding.py'. 

<img width="551" alt="image" src="https://github.com/user-attachments/assets/e828db9c-8897-4f92-a1fe-9d9ffb50ba81">

where pos is the position and i is the dimension.

