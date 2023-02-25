using BioSequences
using IterTools
using FASTX

const letters = [DNA_A,DNA_C,DNA_G,DNA_T]

function read_seqs(file;
                  rna=false)
    ids = String[]
    seqs = LongNuc[]
    FASTAReader(open(file)) do reader
        for record in reader
            seq = LongDNA{4}(String(sequence(record)))
            push!(ids,identifier(record))  
            
            seq = rna ? convert(LongRNA{4},seq) : seq
            push!(seqs,seq)
        end
    end

    return ids,seqs


end

"""
    kmers(k,letters)

Generates the set of unique k-mers for DNA

"""
function kmers(k::Int;seqtype::Type{T} = DNA,
                noambig=true) where T <: NucleicAcid
    letters =  noambig ? [i for i ∈ alphabet(seqtype) if ~isambiguous(i)
                             && i != DNA_Gap && i != RNA_Gap] : alphabet(seqtype)

    x = [letters for i ∈ 1:k]
    kmers = [i for i ∈ product(x...)]

    @assert length(kmers) == 4^k
    return reduce(vcat,kmers)
end


"""
    num_kmers

Returns the number of kmers in a DNA sequence

"""
function num_kmers(x::LongNuc,k::Int)
    return length(x)-k+1
end 


"""
    sequence_to_kmers(x,k)

Converts a sequence x to a sequence of overlapping k-mers

"""
function sequence_to_kmers(x::LongNuc,k::Int;tup=true)
    L = num_kmers(x,k)
    X = Vector{NTuple{k,DNA}}(undef,L)
    for i ∈ 1:L
        X[i] = Tuple(j for j in x[i:i+k-1])
    end
    return X 
end



"""
    count_kmers(x,k)

Converts a sequence x to a sequence of overlapping k-mers

"""
function count_kmers(seqs::Vector{T},k::Int;nuctype=DNA) where T <: LongNuc
    X = zeros(Float32,4^k,length(seqs))
    km = kmer_map(4,nuctype)
    for (j,seq) ∈ enumerate(seqs)
        L = num_kmers(seq,k)
        for i ∈ 1:L
            curr_kmer = seq[i:i+k-1]
            idx = km[curr_kmer]
            X[idx,j] += 1 / L
        end
    end 
    return X 
end


function kmer_map(k::Int,nuctype::Type{T}) where T <: NucleicAcid
    if nuctype == DNA
        f = LongDNA{4}
    elseif nuctype == RNA
        f = LongRNA{4}
    else
        ErrorException
    end 

    Dict(zip(f.(join.(kmers(k,seqtype=nuctype))),1:4^k))
end
