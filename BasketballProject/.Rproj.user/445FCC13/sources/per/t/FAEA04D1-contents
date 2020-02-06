#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#



library(shiny)
library(tidyverse)
library(plotly)
library(RColorBrewer)




data <- read.csv("https://www.dropbox.com/s/u3oijk3eh1oxm3q/BasketballData.csv?dl=1")
data <- data %>%
  select(Rk:X3P_Percent) %>%
  rename(three_p_made = X3P_Made,
         three_p_attempted = X3P_Attempted, 
         three_p_percent = X3P_Percent)

font_settings <- list(
  family = "Courier New, monospace",
  size = 18,
  color = "#7f7f7f"
)
x_settings <- list(
  title = "3PM",
  titlefont = font_settings,
  range=c(50, 400)
)
y_settings <- list(
  title = "3P%",
  titlefont = font_settings,
  range=c(0.30,0.50)
)

league_avgs <- data %>%
  filter(three_p_made > 81) %>%
  summarize(avg_3pm = mean(three_p_made),
            avg_3p_percent = mean(three_p_percent))

annotations_settings <- list(
  x = league_avgs$avg_3pm,
  y = league_avgs$avg_3p_percent,
  text = "League Average",
  xref = "x",
  yref = "y",
  showarrow = TRUE,
  arrowhead = 7,
  ax = 20,
  ay = -40
)

plot1_data <- data %>%
  top_n(30, three_p_attempted)

plot2_data <- data %>%
  filter(Tm != "TOT") %>%
  group_by(Tm) %>%
  summarize(three_p_made = max(three_p_made)) %>%
  inner_join(data, by=c("Tm", "three_p_made"))

plot3_data <- data %>%
  filter(three_p_made >= 82) %>%
  top_n(30, three_p_percent)

plot4_data <- data %>% 
  filter(Tm != "TOT", three_p_made >= 82) %>%
  group_by(Tm) %>%
  summarize(three_p_percent = max(three_p_percent)) %>%
  inner_join(data, by=c("Tm", "three_p_percent"))

p1 <- plot_ly(type = 'scatter', mode = 'markers') %>%
  add_trace(
    x = plot1_data$three_p_made, 
    y = plot1_data$three_p_percent,
    text = plot1_data$Player,
    hoverinfo = 'text',
    marker = list(color='green'),
    showlegend = FALSE
  ) %>%
  layout(title="3PM and 3P% for the Leaders in 3PM", xaxis=x_settings, yaxis=y_settings,
         annotations=annotations_settings)

p2 <- plot_ly(type = 'scatter', mode = 'markers') %>%
  add_trace(
    x = plot2_data$three_p_made, 
    y = plot2_data$three_p_percent,
    text = ~paste(plot2_data$Tm,
                  "<br>", plot2_data$Player),
    hoverinfo = 'text',
    marker = list(color='green'),
    showlegend = FALSE
  ) %>%
  layout(title="3PM and 3P% for Each Team's Leader(s) in 3PM", xaxis=x_settings, yaxis=y_settings,
         annotations=annotations_settings)

p3 <- plot_ly(type = 'scatter', mode = 'markers') %>%
  add_trace(
    x = plot3_data$three_p_made, 
    y = plot3_data$three_p_percent,
    text = plot3_data$Player,
    hoverinfo = 'text',
    marker = list(color='green'),
    showlegend = FALSE
  ) %>%
  layout(title="3PM and 3P% for the Leaders in 3P%", xaxis=x_settings, yaxis=y_settings,
         annotations=annotations_settings)

p4 <- plot_ly(type = 'scatter', mode = 'markers') %>%
  add_trace(
    x = plot4_data$three_p_made, 
    y = plot4_data$three_p_percent,
    text = ~paste(plot4_data$Tm,
                  "<br>", plot4_data$Player),
    hoverinfo = 'text',
    marker = list(color='green'),
    showlegend = FALSE
  ) %>%
  layout(title="3PM and 3P% for Each Team's Leader(s) in 3P%", xaxis=x_settings, yaxis=y_settings,
         annotations=annotations_settings)

plot_list <- list(p1, p2, p3, p4)
all_data <- read.csv("https://www.dropbox.com/s/ebj4qfc2lru2u4e/game_data.csv?dl=1")
rb_data <- read.csv("https://www.dropbox.com/s/qgonmh9bgvjlbhz/rb_data.csv?dl=1") %>%
  select(-X)

# Define server logic required to plot the data
shinyServer(function(input, output) {
  
  plot_num <- reactive({
    case_when(
    input$who_to_display == "Top 30 in 3PM" ~ 1,
    
    input$who_to_display == "Top in 3PM of each team" ~ 2,
    
    input$who_to_display == "Top 30 in 3P%" ~ 3,
    
    input$who_to_display == "Top in 3P% of each team" ~ 4
  )})
  
  output$my_plot <- renderPlotly({plot_list[[plot_num()]]})
  
  show_teams <- reactive({input$teams})
  
  show_games <- reactive({input$games_played})
  
  color_pallete <- brewer.pal(8, "Dark2")
  
  output$wl_plot <- renderPlotly({all_data %>%
    filter(team %in% show_teams(),
           G <= show_games()) %>%
    plot_ly(x=~G, y=~win_loss,
            type="scatter", mode="lines",
            text = ~paste(team,
                          "<br>", "W/L: ", round(win_loss, digits=2),
                          "<br>", "Gm: ", G),
            hoverinfo='text', color=~team,
            colors=color_pallete) %>%
    layout(title="Game-to-Game Win/Loss Ratio\n(as of 12/14/19)",
           xaxis=list(
             title = "Games Played",
             titlefont = font_settings,
             range = c(0, max(all_data$G))),
           yaxis=list(
             title = "Win/Loss Ratio",
             titlefont = font_settings,
             range = c(0, max(all_data$win_loss, na.rm=TRUE)))
           )})
  
  show_players <- reactive({input$players})
  
  show_games_rb <- reactive({input$games_played_rb})
  
  color_pallete2 <- brewer.pal(10, "Dark2")
  
  output$rb_plot <- renderPlotly({rb_data %>%
    filter(player %in% show_players(),
           game <= show_games_rb()) %>%
    plot_ly(x=~game, y=~sum_rb, 
            type="scatter",
            mode="lines",
            text = ~paste(player,
                          "<br>", "TRB: ", sum_rb,
                          "<br>", "Gm: ", game),
            hoverinfo='text',
            color=~player) %>%
    layout(title="Total Rebounds for the Top 10 Rebounders in the NBA\n(as of 12/16/19)",
           xaxis=list(
             title = "Games Played",
             titlefont = font_settings,
             range = c(1, max(rb_data$game))),
           yaxis=list(
             title = "Total Rebounds",
             titlefont = font_settings,
             range = c(0, max(rb_data$sum_rb) + 50))
           )
    })
  
})
