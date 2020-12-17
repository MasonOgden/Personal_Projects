# reading in lines
cd("C:/Users/mogde/Desktop/Git Repositories/Personal_Projects/AdventOfCode/Day6")

lines = open("input.txt") do f
    readlines(f)
end

# processing lines
line_num = 1
all_info = []

while line_num <= size(lines)[1]
    this_group_set = Set()
    while (line_num <= size(lines)[1]) and lines[line_num] != "\n"
        
    end

end
