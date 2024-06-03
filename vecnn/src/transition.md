# Tadeo: 2024-06-01 7:34 

transition steps: 


1. build a vp-tree.
2. get all the areas of the vp-tree on a certain level,
the subtrees remaining should be roughly the same size. e.g. we split to have 16-32 nodes per "chunk".
the root node of a subtree is part of the left side when split.

3. for each chunk, we write the chunks nodes into the lowest level of an hnsw structure.
4. we insert connections between all nodes of the chunk. the chunk size should be smaller than the max number of neihgbors held in the hnsw. These are chunksize*chunksize/2 distance calculations. Write all of them into an array, then write the neighbors into the binary heaps of all the nodes.

now all nodes in a chunk are connected to each other, we now need to add connections between chunks. 
lets look at 8 chunks at a time. 
for each of these pairs, we want to try to insert a connection.
By that we mean one connection between a pair of points where one point is in the other chunk and one point is in this chunk.

We can do a little dance: 
pick a random representative a from chunk 1 and b from chunk 2.
search from a to b in chunk 1 -> point c
search from b to a in chunk 2 -> point d
add a connection between them, replacing the worst connection 

-> keep track of how many times this leads to a better connection.

if we do this between 8 random chunks, we probably get no good connections at all?


