module Attention

open Types
open HelperFunctions

// Computes attention for a single token.
// Equivalent to the innermost loop of transformer() in the C# implementation.
// i.e. compute the square root of the dot product between query and key vectors.
// Hint: Use the keyLookup function, as we do not have the key vector directly here.
let attentionScore (query:Vector) (keyLookup:int->float) : float =    
    // TODO: Implement this function.
    query
        |> Array.mapi (fun i q -> q * keyLookup i)
        |> Array.sum
        |> fun i -> i / System.Math.Sqrt query.Length
    // Scale by square root of vector dimension
    //raise (System.NotImplementedException("Attention attentionScore not implemented"))

// Compute the dot product of the attention vector with the value vector.
let weightedAttention (attention: Vector) (valueLookup:int->float) : float =
    // TODO: Implement this function.
    attention
    |> Array.mapi (fun i a -> a * valueLookup i)
    |> Array.sum
    //raise (System.NotImplementedException("Attention weightedAttention not implemented"))

// Computes attention for one head of multi-head attention, using the query, key and value vectors.
// This is equivalent to the n_heads loop in the transformer() function in the C# implementation.    
let attentionForOneHead (keyLookup:int->int->float) (valueLookup:int->int->float) (tokenPosition:int) (query: Vector): Vector =
    // TODO: Implement this function.
    //// Compute attention scores for each token
    let scores = 
        [|0 .. tokenPosition|]
        |> Array.map (fun i -> attentionScore query (fun j -> keyLookup i j))
    //Compute softmax weights
    let weights = softMax scores

    // Compute the output vector using weighted sum of value vectors
    let outputVector = 
        Array.init (Array.length query) (fun j ->
            [|0 .. tokenPosition|]
            |> Array.map (fun i -> (valueLookup i j) * weights.[i])
            |> Array.sum)
    outputVector
    //raise (System.NotImplementedException("Attention attentionForOneHead not implemented")) 

// Computes attention for all heads in multi-head attention.
// Hint: Instead of returning multiple vectors, one for each head, this array should be flattened with flattenMultipleHeads().
let attention (keyLookup:int->int->int->float) (valueLookup:int->int->int->float) (tokenPosition:int) (query: MultiHead) : Vector =
    // TODO: Implement this function
    // Compute the attention for each head
    [| 0 .. (Array.length query) - 1 |]  // Iterate over the number of heads
    |> Array.map (fun headIndex -> 
        // For each head, compute the attention for all tokens using query[headIndex]
        attentionForOneHead 
            (fun i -> keyLookup headIndex i)    // Key lookup for this head
            (fun i -> valueLookup headIndex i)  // Value lookup for this head
            tokenPosition 
            query.[headIndex]                   // The query vector for this head
    )
    |> flattenMultipleHeads
    //Mapi 2 liner pipe #Possible
    //raise (System.NotImplementedException("Attention attention not implemented"))