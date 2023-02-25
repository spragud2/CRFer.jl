using BioSequences
using IterTools
using FASTX

const letters = [DNA_A,DNA_C,DNA_G,DNA_T]

function read_seqs(file;
                  rna=false)
    seq_d = Dict{String,LongNuc}()
    FASTAReader(open(file)) do reader
        for record in reader
            seq = LongDNA{4}(String(sequence(record)))
            seq_d[identifier(record)] = rna ? convert(LongRNA{4},seq) : seq
        end
    end

    return seq_d


end

"""
    kmers(k,letters)

Generates the set of unique k-mers for DNA

"""
function kmers(k::Int,letters::Vector{DNA}=letters)
    x = [letters for i ∈ 1:k]
    kmers = [i for i ∈ product(x...)]
    return reduce(vcat,kmers)
end

"""
    num_kmers

Returns the number of kmers in a DNA sequence

"""
function num_kmers(x::LongSequence{DNAAlphabet{4}},k::Int)
    return length(x)-k+1
end 


"""
    sequence_to_kmers(x,k)

Converts a sequence x to a sequence of overlapping k-mers

"""
function sequence_to_kmers(x::LongSequence{DNAAlphabet{4}},k::Int)
    L = num_kmers(x,k)
    println(L)
    X = Vector{NTuple{k,DNA}}(undef,L)
    for i ∈ 1:L
        X[i] = Tuple(j for j ∈ x[i:i+k-1])
    end
    return X 
end

