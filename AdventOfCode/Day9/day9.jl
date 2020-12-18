cd("C:/Users/mogde/Desktop/git_repos/Personal_Projects/AdventOfCode/Day9")

function read_num_file(filename, preamble_length = 25)
    lines = open(filename) do f
        parse.(Int64, readlines(f))
    end

    preamble = lines[begin:preamble_length]
    nums = lines[preamble_length + 1: end]

    return preamble, nums
end

test_preamble, test_nums = read_num_file("test_input.txt", 5)
all_test_nums = vcat(test_preamble, test_nums)

function find_first_invalid_num(preamble, nums)
    len_preamble = size(preamble)[1] # number of nums in the preamble
    all_nums = vcat(preamble, nums) # all numbers in one vector

    for i in (len_preamble + 1):size(all_nums)[1] # for every index in the non-preamble nums
        num_to_be_tested = all_nums[i]
        if !is_valid(all_nums[(i - len_preamble): i - 1], num_to_be_tested)
            return num_to_be_tested
        end
    end
end



function is_valid(before_nums, test_num)
    combo_sums = Set{Int64}(sum.(generate_combinations(before_nums)))
    in(test_num, combo_sums)
end

function generate_combinations(input_array)
    list_w_sames = collect(Set{Vector{Int64}}([sort([a, b]) for a in input_array for b in input_array]))
    filter((pair) -> pair[1] != pair[2], list_w_sames)
end

#find_first_invalid_num(test_preamble, test_nums)

preamble, nums = read_num_file("input.txt")
all_nums = vcat(preamble, nums)

invalid_num = find_first_invalid_num(preamble, nums)

function find_contiguous_set(all_nums, sum_number)
    for i in 1:(size(all_nums)[1])
        this_sum = 0
        these_numbers = Vector{Int64}()
        position = i
        while this_sum < sum_number # while we haven't gone over the number
            this_sum += all_nums[position]
            push!(these_numbers, all_nums[position])
            position += 1
        end

        if (length(these_numbers) >= 2) && (this_sum == sum_number)
            return minimum(collect(these_numbers)), maximum(collect(these_numbers))
        end
    end
end

find_contiguous_set(all_test_nums, 127)

min_max = find_contiguous_set(all_nums, invalid_num)

sum(min_max)
