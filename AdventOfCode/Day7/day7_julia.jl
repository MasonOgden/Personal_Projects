cd("C:/Users/mogde/Desktop/git_repos/Personal_Projects/AdventOfCode/Day7")

get_group(regexmatch) = regexmatch[1]

function extract_info(inside_rule)
    this_match = match(r"(\d+) ([\w\s]+)", inside_rule)
    return parse(Int64, this_match[1]), this_match[2]
end


function append_rule!(rule_dictionary, rule_string)
    beginning_match = match(r"(^[\w\s]+) bags contain ", rule_string)
    outside_bag_color = beginning_match[1]
    inside_strings = replace(rule_string, beginning_match.match => "")
    all_inside = get_group.(collect(eachmatch(r"([no|\d]+[\w\s]+) bag", inside_strings)))
    if all_inside[1] == "no other"
        rule_dictionary[outside_bag_color] = nothing
    else
        inside_dict = Dict{String, Int64}()
        for inside_rule in all_inside
            info = extract_info(inside_rule)
            inside_dict[info[2]] = info[1]
        end
        rule_dictionary[outside_bag_color] = inside_dict
    end
end

function parse_rule_file(filename)
    lines = open(filename) do f
        readlines(f)
    end

    rule_dict = Dict()

    for line in lines
        append_rule!(rule_dict, line)
    end

    rule_dict
end

function find_bags(rules_dict, inserting)
    bag_set = Set{String}()
    # finding bags that directly contain the inserting bag
    for key in collect(keys(rules_dict))
        if (rules_dict[key] != nothing) && haskey(rules_dict[key], inserting)
            push!(bag_set, key)
        end
    end

    added = length(bag_set)
    #println("Before while loop, added = $added")
    # while we are still finding bags that contain bags that contain ... shiny gold
    while added > 0
        added = 0 # reset counter
        for inside_color in collect(bag_set) # for every color that directly/indirectly contains shiny gold
            for outside_color in collect(keys(rules_dict)) # for every color in the rules
                if (rules_dict[outside_color] != nothing) && haskey(rules_dict[outside_color], inside_color)
                    if !in(outside_color, bag_set)
                        #println("Adding $outside_color")
                        push!(bag_set, outside_color)
                        added += 1
                    end
                end
            end
        end
    end
    bag_set
end

#test_rules = parse_rule_file("test_input.txt")

#inserting = "shiny gold"

#test_result = find_bags(test_rules, inserting)
#test_answer = length(test_result)
#println("Correct answer to the test case: 4. My answer: $test_answer")

#rules = parse_rule_file("input.txt")

#result = find_bags(rules, "shiny gold")
#answer = length(result)

#println("Answer to part 1: $answer")

struct Node
    color::String
    parent::Node
    parent_weight::Int64
    children::Dict{Int64, Node}
end

color(n::Node) = n.name
parent_node(n::Node) = n.parent
parent_weight(n::Node) = n.parent_weight
children_nodes(n::Node) = n.children


function count_inner_bags(rules_dict, outside_bag_color)
    result = recursive_count(rules_dict, outside_bag_color)
    result
end

function recursive_count(rules_dict, color)
    # set of children that are not leaves
    non_leaf_colors = Set{String}()
    for key in collect(keys(rules_dict[color]))
        if !isnothing(rules_dict[key]) # if this color is not a leaf
            push!(non_leaf_colors, key)
        end
    end
    if length(non_leaf_colors) == 0 # if we are at a node that has only leaves
        return sum(collect(values(rules_dict[color])))
    else # if we still have branches
        # then call this function on all those branches with the weightings
        result = 0
        for key in collect(non_leaf_colors)
            new_multiplier = rules_dict[color][key]
            addition = new_multiplier + (new_multiplier * recursive_count(rules_dict, key))
            result += addition
        end
        return result
    end
end

test_rules2 = parse_rule_file("test_input2.txt")

p2_test_result = count_inner_bags(test_rules, "shiny gold")
p2_test_result2 = count_inner_bags(test_rules2, "shiny gold")

p2_results = count_inner_bags(rules, "shiny gold")

count_inner_bags(test_rules2, "shiny gold")
