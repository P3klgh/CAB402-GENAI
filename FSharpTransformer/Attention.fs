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
    // Determine the size of the query vector (also the size of the key and value vectors)
    let headSize = query.Length
    // Array to hold the attention scores for each token (from 0 to tokenPosition)
    let attentionScores = Array.zeroCreate<float> (tokenPosition + 1)
    let scores = [| for i in 0 .. tokenPosition -> attentionScore query (fun j -> keyLookup i j ) |]
    let weights = softMax scores
    let outputVector = [| for j in 0 .. (Array.length query) - 1 -> Array.sum [| for i in 0 .. tokenPosition -> (valueLookup i j) * weights.[i] |] |]
    //let outputVector = 
    //    query
    //        |> Array.mapi (fun j _ ->
    //            [| 0 .. tokenPosition |]
    //        |> Array.mapi (fun i _ -> valueLookup i j * weights.[i])
    //        |> Array.sum
    //)
    outputVector
    //raise (System.NotImplementedException("Attention attentionForOneHead not implemented")) 

    //raise (System.NotImplementedException("Attention attentionForOneHead not implemented"))

// Computes attention for all heads in multi-head attention.
// Hint: Instead of returning multiple vectors, one for each head, this array should be flattened with flattenMultipleHeads().
let attention (keyLookup:int->int->int->float) (valueLookup:int->int->int->float) (tokenPosition:int) (query: MultiHead) : Vector =
    // TODO: Implement this function
    raise (System.NotImplementedException("Attention attention not implemented"))
    //[| for i in 0 .. tokenPosition -> attentionForOneHead (fun j -> keyLookup i j) (fun j -> valueLookup i j) i qu
    //ery |]
    //    |> Array.concat