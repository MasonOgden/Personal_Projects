cd("C:/Users/mogde/Desktop/git_repos/Personal_Projects/AdventOfCode/Day9")


function read_initial_state_file(filename)
    lines = open(filename) do f
        readlines(f)
    end

    return Dict{Int64,Vector{String}}(0 => lines)
end

test_init = read_initial_state_file("test_input.txt")
