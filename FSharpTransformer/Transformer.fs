module Transformer

open Types
open IO
open HelperFunctions
open Attention



// Return a new function that takes the head number, requested position, and position within head, returning the key/value
// from [head number][postion within head] in either the newValues matrix if the token position is the current position,
// otherwise in the history/cache table at [requested position][layer].
let createLookupFunction (previousValues:MultiHead[][]) (newValues:MultiHead) (tokenPosition:int) (layer:int): (int -> int -> int -> float) =
    fun headNumber requestedPosition positionWithinHead ->
        // TODO: Implement this function.
        match requestedPosition with
        | pos when pos = tokenPosition -> newValues.[headNumber].[positionWithinHead]
        | pos -> previousValues.[pos].[layer].[headNumber].[positionWithinHead]
        //if requestedPosition = tokenPosition then 
        //    newValues.[headNumber].[positionWithinHead]
        //else
        //    previousValues.[requestedPosition].[layer].[headNumber].[positionWithinHead]
        //raise (System.NotImplementedException("Transformer createLookupFunction not implemented"))

// Processes one layer of the transformer model. This is equivalent to the first for loop in the C# transformer() function.
// The parameters you will need are stored in the model.weights array under index layer.
// You need to:
// - 1. Apply layer normalization to the current vector before attention using normalizeInputWeights
// - 2. Generate query, key and value vectors by multiplying the current vector by the corresponding query (wq), key (qk) and value (wv)
//   matrices for this layer. You will need to use the reshapeToMultipleHeads() function to split these vectors.
// - 3. Apply Rotary Position Embedding(RoPE) to query and key vectors. The value vector is not rotated.
// - 4. Use the attention function to compute multi-head attention for the query/key/value vectors.
// - 5. Project concatenated attention outputs with the output matrix (wo) to produce final attention.
// - 6. Add the residual connection (input vector).
// - 7. Apply layer normalization before the final feed-forward neural network (normalizeAttentionWeights).
// - 8. Feed-forward network component: Matrix multiply w1 and w3, sigmoid is only applied to w1.
// - 9. Then the product of these two matrices is multiplied by w2 with second residual connection.
let feedforwardOneLayer (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (input: Vector) (layer: int) : Vector * MultiHead * MultiHead =
    // TODO: Implement this function.
    let weights = model.weights.[layer]

    // 1. Layer normalization before attention
    let normalisedInput = rootMeanSquareNormalize weights.normalizeInputWeights input

    // 2. Linear projections to Q/K/V
    let q = matrixMultiply weights.wq normalisedInput
    let k = matrixMultiply weights.wk normalisedInput
    let v = matrixMultiply weights.wv normalisedInput

    // 3.1 Reshape to heads
    let qHeads = reshapeToMultipleHeads model.headSize q
    let kHeads = reshapeToMultipleHeads model.headSize k
    let vHeads = reshapeToMultipleHeads model.headSize v

    // 3.2 Apply Rotary Position Embedding (RoPE) to Q/K
    let rope = model.rotationCoefficients.[tokenPosition]
    let qRotated = rotateVector rope qHeads
    let kRotated = rotateVector rope kHeads

    // Define key/value lookup functions (for caching past tokens)
    let keyLookup = createLookupFunction keyCache kRotated tokenPosition layer
    let valueLookup = createLookupFunction valueCache vHeads tokenPosition layer

    // 4. Multi-head attention using Q, K, V, and cache
    let attentionOutput = attention keyLookup valueLookup tokenPosition qRotated

    // 5 Flatten heads and apply output projection
    let attentionConcat = attentionOutput
    let projectedAttention = matrixMultiply weights.wo attentionConcat

    // 6. First residual connection (post-attention)
    let attentionWithResidual = add input projectedAttention 

    // 7. Normalize before feed-forward
    let normalizedAttention = rootMeanSquareNormalize weights.normalizeAttentionWeights attentionWithResidual

    // 8. Feed-forward block
    let w1 = matrixMultiply weights.w1 normalizedAttention
    let w3 = matrixMultiply weights.w3 normalizedAttention
    let gated = elementWiseMultiply (sigmoidActivation w1) w3
    let ffOutput = matrixMultiply weights.w2 gated

    // 9. Second residual connection (final output)
    let finalOutput = add ffOutput attentionWithResidual

    // Return final output and updated key/value heads for caching
    finalOutput, kHeads, vHeads
    
    //raise (System.NotImplementedException("Transformer feedforwardOneLayer not implemented"))

// Returns a new array with the newElement added to array.
let appendElement (array: 'T[]) (newElement: 'T) : 'T[] =
    Array.append array [| newElement |]

// Feeds an input vector through all layers of the transformer.
// This function is also responsible for updating the key/value cache that is used to retrieve the vectors in later layers.
// The cache is "updated" for each layer by appending to the end of the array representing the cache.
let feedForwardAllLayers (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (input:Vector)  : Vector * MultiHead[] * MultiHead[] =
    // Use List.fold to process each layer, accumulating the input with each.
    let Folder (input, previousKeys, previousValues) layer =
        let (output, keys, values) = feedforwardOneLayer model keyCache valueCache tokenPosition input layer
        (output, appendElement previousKeys keys,  appendElement previousValues values)
    List.fold Folder (input, Array.empty, Array.empty) [0 .. model.numberOfLayers-1]

// Uses all the transformer model's layers to predict the next token that follows token.
// The output is the logits for each token, and the key/value cache for all layers for this token.
// This function roughly equates to the first copy() call and final rmsnorm()/matmul() calls in the C# transformer() method.
let feedForward (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (token:Token) : Vector * MultiHead[] * MultiHead[] =
    // TODO: Implement this function.
    // Convert token to input vector using token embedding
    let input = model.tokenEmbedding.[token]

    // Process input through all layers of the transformer
    let (finalOutput, updatedKeyCache, updatedValueCache) = feedForwardAllLayers model keyCache valueCache tokenPosition input

    // Apply final RMS normalization using model.normalizeOutputWeights
    let normalized = rootMeanSquareNormalize model.normalizeOutputWeights finalOutput

    // Project to logits using the output projection matrix from the last layer
    let logits = matrixMultiply model.weights.[model.numberOfLayers - 1].wo normalized

    // Return logits and updated key/value cache
    logits, updatedKeyCache, updatedValueCache
    //raise (System.NotImplementedException("Transformer feedForward not implemented"))

// Obtains the logits for the next token, and selects the token to return based on the provided decoder function.
// You should also return the updated key/value cache.
let generateNextToken (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][])  (tokenPosition:int) (token:Token) (decoder:Vector->Token) : Token * MultiHead[] * MultiHead[] =
    // TODO: Implement this function.
    // Run the transformer forward pass
    let (logits, updatedKeyCache, updatedValueCache) = feedForward model keyCache valueCache tokenPosition token
    
    // Decode the logits to select the next token
    let nextToken = decoder logits
    
    // Return token and updated key/value caches
    nextToken, updatedKeyCache, updatedValueCache
    //raise (System.NotImplementedException("Transformer generateNextToken not implemented"))

// Generates a sequence of tokens using the specified decoder.
// This function is responsible for appending the cache of key/values for all layers to the "main" key/value cache,
// which contains the key/values for all layers of every preceding token.
// The start and end of the sequence are indicated by the token 1, therefore we should stop producing tokens after
// a token of 1 is predicted. Each token is also printed out as it is generated.
let generateTokenSequence (model: Model) (decoder:Vector->Token) : string seq = 
    (1, 0, Array.empty, Array.empty) 
    |> Seq.unfold (fun (token, tokenPosition, previousKeys, previousValues) -> 
        let (nextToken, keys, values) = generateNextToken model previousKeys previousValues tokenPosition token decoder
        let newKeys = appendElement previousKeys keys
        let newValues = appendElement previousValues values
        if nextToken = 1 || tokenPosition+1 = model.seqenceLength
        then None
        else
            Some (model.vocabulary.[nextToken], (nextToken, tokenPosition+1, newKeys, newValues)))

let tellStory (model: Model) (decoder:Vector->Token) : unit =
    generateTokenSequence model decoder
    |> printStory