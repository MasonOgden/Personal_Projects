cd("C:/Users/mogde/Desktop/Git Repositories/Personal_Projects/AdventOfCode/Day10")

function read_adapter_file(filename)
    lines = open(filename) do f
        parse.(Int64, readlines(f))
    end
end

function find_adapter_sequence(adapter_list)
    slist = sort(adapter_list)

    current_voltage = 0
    used_adapters = Vector{Int64}()
    diff_dict = Dict{Int64, Int64}(1 => 0, 2 => 0, 3 => 0)

    while (size(used_adapters)[1] < size(slist)[1]) # while we haven't
        candidates = find_candidate_adapters(slist, current_voltage)
        min_candidate = minimum(candidates)
        diff_dict[(min_candidate - current_voltage)] += 1
        current_voltage = min_candidate
        push!(used_adapters, min_candidate)
    end

    diff_dict[3] += 1
    return diff_dict
end

function find_candidate_adapters(sorted_adapter_list, current_voltage)
    candidates = Vector{Any}()
    for num in sorted_adapter_list
        if 1 <= (num - current_voltage) <= 3
            push!(candidates, num)
        end
    end
    candidates
end

list1 = read_adapter_file("test_input1.txt")

#list2 = read_adapter_file("test_input2.txt")

#list3 = read_adapter_file("input.txt")

# what if I generated a tree of possible arrangements?
# the root would be (1), with one child, (4), then
# (4) would have 3 children: (5, 6, 7)

mutable struct Node
    parent::Any
    voltage::Int64
    children::Vector{Any}
end

parent_node(n::Node) = n.parent
voltage(n::Node) = n.voltage
children(n::Node) = n.children

function possible_parents(slist, voltage)
    possible_list = Vector{Int64}()

    for num in slist
        if 1 <= (voltage - num) <= 3
            push!(possible_list, num)
        end
    end

    if isempty(possible_list)
        return [0]
    else
        return possible_list
    end
end

function find_node(node_list, voltage)
    for node in node_list
        if node.voltage == voltage
            return node
        end
    end
end

function generate_arrangements(adapter_list)
    slist = sort(adapter_list)

    current_voltage = 0

    created_nodes = Vector{Node}()

    # Creating a node for each adapter
    for voltage in adapter_list
        parents = possible_parents(slist, voltage)
        children = find_candidate_adapters(slist, voltage)
        this_node = Node(parents, voltage, children)
        push!(created_nodes, this_node)
    end

    sort!(created_nodes, by=voltage, rev=true)

    for node in created_nodes
        if parent_node(node)[1] == 0
            node.parent[i] = nothing
            root_node = node
        else
            parent_voltages = parent_node(node)
            for i in 1:(size(parent_voltages)[1])
                parent_voltage = parent_node(node)[i]
                node.parent[i] = find_node(created_nodes, parent_voltage)
            end
        end
    end
end

list1

test = generate_arrangements(list1)
