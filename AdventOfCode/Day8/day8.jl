cd("C:/Users/mogde/Desktop/git_repos/Personal_Projects/AdventOfCode/Day8")

function read_boot_file(filename)
    lines = open(filename) do f
        readlines(f)
    end

    convert_item_to_int(pair) = [pair[1], parse(Int64, pair[2])]

    items = convert_item_to_int.(split.(lines, " "))

    items
end

function find_acc_before_loop(commands)
    position = 1
    accumulator = 0

    visited_positions = Set{Int64}()

    while position <= size(commands)[1]
        if in(position, visited_positions)
            return Dict{String,Any}(
                "result" => "infinite loop",
                "accumulator value" => accumulator,
            )
        end
        if commands[position][1] == "jmp"
            push!(visited_positions, position)
            position += commands[position][2]
        elseif commands[position][1] == "acc"
            push!(visited_positions, position)
            accumulator += commands[position][2]
            position += 1
        else
            push!(visited_positions, position)
            position += 1
        end
    end
    Dict{String,Any}("result" => "completed", "accumulator value" => accumulator)
end

test_commands = read_boot_file("test_input.txt")

#find_acc_before_loop(test_commands)

# answer to part 1
real_commands = read_boot_file("input.txt")

#find_acc_before_loop(real_commands)

# function for part 2

#test_commands_no_loop = read_boot_file("test_input_correct.txt")
#find_acc_before_loop(test_commands_no_loop)

function swap_command!(commands, index)
    if (commands[index][1] == "jmp")
        commands[index][1] = "nop"
    else
        commands[index][1] = "jmp"
    end
end

function find_and_change_error(commands)
    for i in 1:(size(commands)[1])
        if (commands[i][1] == "jmp") | (commands[i][1] == "nop")
            println("Changing:", commands[i])
            swap_command!(commands, i) # swap jmp for nop or vice versa
            result = find_acc_before_loop(commands)
            swap_command!(commands, i) # swap them back
            if (result["result"] == "completed")
                return result["accumulator value"]
            end
        end

    end
end

find_and_change_error(test_commands)

find_and_change_error(real_commands)
