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
   
    let weights = model.weights.[layer]
    let rope = model.rotationCoefficients.[tokenPosition]

    // Attention block
    let normalizedInput =
        input
        |> rootMeanSquareNormalize weights.normalizeInputWeights

    let q, k, v =
        normalizedInput
        |> fun x -> matrixMultiply weights.wq x,
                    matrixMultiply weights.wk x,
                    matrixMultiply weights.wv x

    let qRotated, kRotated, vHeads =
        let reshape = reshapeToMultipleHeads model.headSize
        reshape q |> rotateVector rope,
        reshape k |> rotateVector rope,
        reshape v

    let keyLookup = createLookupFunction keyCache kRotated tokenPosition layer
    let valueLookup = createLookupFunction valueCache vHeads tokenPosition layer

    let attentionOutput =
        qRotated
        |> attention keyLookup valueLookup tokenPosition

    let projectedAttention =
        attentionOutput
        |> matrixMultiply weights.wo

    let attentionWithResidual =
        input
        |> add projectedAttention

    // Feed-forward block
    let ffOutput =
        attentionWithResidual
        |> rootMeanSquareNormalize weights.normalizeAttentionWeights
        |> fun x ->
            let w1 = matrixMultiply weights.w1 x
            let w3 = matrixMultiply weights.w3 x
            sigmoidActivation w1
            |> elementWiseMultiply w3
            |> matrixMultiply weights.w2

    let finalOutput =
        ffOutput
        |> add attentionWithResidual

    finalOutput, kRotated, vHeads
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
    // Convert token to input vector
    let input = model.tokenEmbedding.[token]

    // Process through transformer layers
    let finalOutput, updatedKeyCache, updatedValueCache =
        input
        |> feedForwardAllLayers model keyCache valueCache tokenPosition

    // Compute logits
    let logits =
        finalOutput
        |> rootMeanSquareNormalize model.normalizeOutputWeights
        |> matrixMultiply model.tokenEmbedding

    logits, updatedKeyCache, updatedValueCache

// Obtains the logits for the next token, and selects the token to return based on the provided decoder function.
// You should also return the updated key/value cache.
let generateNextToken (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][])  (tokenPosition:int) (token:Token) (decoder:Vector->Token) : Token * MultiHead[] * MultiHead[] =
    // TODO: Implement this function.
    // Run the transformer forward pass and decode next token
    let logits, updatedKeyCache, updatedValueCache =
        feedForward model keyCache valueCache tokenPosition token

    let nextToken =
        logits
        |> decoder

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