% ====== Expressiveness and Well-Formedness Constraints ======

% === Within Encodings ===

% = new ones start =

% @constraint Don't use row with quantitative (added)
hard(row_no_quant, E) :- channel(E,row), type(E,quantitative).
hard(col_no_quant, E) :- channel(E,column), type(E,quantitative).

hard(col_not_too_big, E) :- channel(E,column), enc_cardinality(E,C), C > 10.
hard(row_not_too_big, E) :- channel(E,row), enc_cardinality(E,C), C > 10.


hard(row_with_high_cardinality,E,C) :- channel(E,row), enc_cardinality(E,C), C > 20.
hard(col_with_high_cardinality,E,C) :- channel(E,col), enc_cardinality(E,C), C > 20.

hard(nominal_with_high_cardinality,E,C) :- type(E,nominal), enc_cardinality(E,C), C > 60.
hard(ordinal_with_high_cardinality,E,C) :- type(E,ordinal), not bin(E,_), not aggregate(E,_), enc_cardinality(E,C), C > 50.

%hard(same_fields) :- field(A,F),channel(A,_),field(B,F),channel(B,_), A != B.
%hard(same_fields) :- field(A,F),channel(A,_),field(B,F),channel(B,_), not hide_axis(A), not hide_axis(B), A != B.
% same field should have same datatype (guaranteed through enumeration )
hard(same_field_same_type) :- field(E1,F),type(E1,T1),field(E2,F),type(E2,T2), E1 != E2, T1 != T2.
hard(multiple_aggregation) :- aggregate(A,_), aggregate(B,_), A != B.

% (added)
hard(temporal_not_in_size, E) :- channel(E,size), type(E,temporal).
hard(temporal_not_in_color, E) :- channel(E,color), type(E,temporal).

% Don't bin if the cardinality is low (added)
hard(bin_low_cardinality,E,C) :- bin(E,_), enc_cardinality(E,C), C < 20.

% Positional encoding's priority is higher than non-positional
hard(only_x_multiple_non_pos) :- channel(_,y), not channel(_,x), {channel(_,C): non_positional(C)} > 1.
hard(only_y_multiple_non_pos) :- channel(_,x), not channel(_,y), {channel(_,C): non_positional(C)} > 1.

% Prefer to use only x instead of only y.
hard(only_x) :- channel(_,y), not channel(_,x).
hard(quant_bin,E) :- bin(E,_), not type(E,quantitative).
hard(only_discrete) :- not continuous(_).

% @constraint Don't use row/column without x/y.
hard(row_no_x) :- channel(_,row), not channel(_,x).
hard(column_no_y) :- channel(_,column), not channel(_,y).
hard(row_no_column) :- channel(_,row), not channel(_,column).

hard(high_cardinality_size,E) :- continuous(E), channel(_,size), enc_cardinality(E,C), C > 50, channel(E,(x;y)).

% @constraint disable using log
hard(no_zero,E) :- zero(E).
hard(no_log,E) :- log(E).

%% normalization (1)line chart and area charts requires y continuous (added)
hard(line_area_x_discrete_y_cont) :- mark(line;area), channel_discrete(y).
hard(line_area_x_discrete_y_cont) :- mark(line;area), channel(E, y), type(E, temporal).

%% don't use high cardinality color or size for line charts (added)
hard(line_area_x_discrete_color_cont) :- mark(line;area), channel_continuous(color).
hard(line_area_x_discrete_color_cont_2) :- mark(line;area), continuous(E), channel(E,color), enc_cardinality(E,C), C > 10.

hard(line_area_x_discrete_size_cont) :- mark(line;area), channel_continuous(size).
hard(line_area_x_discrete_size_cont_2) :- mark(line;area), continuous(E), channel(E,size), enc_cardinality(E,C), C > 10.

%% bar chart should not have x continous (to break symmetry) (added)
hard(bar_x_not_continous) :- mark(bar), channel_continuous(x).

hard(line_x_not_continous) :- mark(line), channel_continuous(x).
hard(line_x_not_discrete) :- mark(line), channel_discrete(x).
hard(line_no_nominal) :- mark(line), channel(x, nominal).
hard(area_x_not_continous) :- mark(area), channel_continuous(x).
hard(area_x_not_discrete) :- mark(area), channel_discrete(x).
hard(area_no_nominal) :- mark(area), channel(x, nominal).

hard(point_no_nominal) :- mark(point), channel(x, nominal).
hard(point_no_nominal) :- mark(point), channel(x, temporal).

%% bar chart must have both x and y axes. (added)
hard(bar_no_x) :- mark(bar), not channel(_,x).
hard(bar_no_y) :- mark(bar), not channel(_,y).

%% avoid count on x, y while having color / size enabled
%hard(not_count_x_with_size) :- channel(E,x), aggregate(E,count), channel(_, size).
%hard(not_count_x_with_color) :- channel(E,x), aggregate(E,count), channel(_, color).
%hard(not_count_y_with_size) :- channel(E,y), aggregate(E,count), channel(_, size).
%hard(not_count_y_with_size) :- channel(E,y), aggregate(E,count), channel(_, color).

% prevent only count on x
%hard(no_only_count_x) :- channel(EX,x), channel(EY,y), aggregate(EX,count), not aggregate(EY,count).

% prevent only binning on y (?)
hard(no_only_bin_y) :- channel(EX,x), channel(EY,y), not bin(EX,_), bin(EY,_).

% @constraint arc require color and theta channel.
hard(arc_without_theta) :- mark(arc), not channel(_,theta).
hard(arc_without_color) :- mark(arc), not channel(_,color).

hard(arc_with_x) :- mark(arc), channel(_,x).
hard(arc_with_y) :- mark(arc), channel(_,y).
hard(arc_with_size) :- mark(arc), channel(_,size).
hard(theta_non_quant) :- channel(E,theta), not type(E,quantitative).

hard(theta_without_arc) :- not mark(arc), channel(_,theta).
hard(aggregte_theta,E) :- channel(E,theta), aggregate(E,_).

% = new ones end =

% @constraint Primitive type has to support data type. (added)
hard(enc_type_valid,E,F) :- type(E,quantitative), field(E,F), fieldtype(F,(string;boolean;datetime)).
hard(enc_type_valid,E,F) :- type(E,temporal), field(E,F), not fieldtype(F,datetime).

% @constraint Can only bin quantitative or ordinal. (added)
hard(bin_q_o,E,T) :- type(E,T), bin(E,_), T != quantitative, T != ordinal.

% @constraint Can only use log with quantitative.
% hard(log_q,E) :- log(E), not type(E,quantitative).

% @constraint Can only use zero with quantitative.
% hard(zero_q,E) :- zero(E), not type(E,quantitative).

% @constraint Cannot use log scale with discrete (which includes binned).
% hard(log_discrete,E) :- log(E), discrete(E).

% @constraint Cannot use log and zero together.
% hard(log_zero,E) :- log(E), zero(E).

% @constraint Cannot use log if the data is negative or zero.
% hard(log_non_positive,E,F) :- log(E), field(E,F), extent(F,MIN,_), MIN <= 0.

% @constraint Cannot bin and aggregate.
hard(bin_and_aggregate,E) :- bin(E,_), aggregate(E,_).

% @constraint Ordinal only supports min, max, and median.
hard(aggregate_o_valid,E,A) :- type(E,ordinal), aggregate(E,A), A != min, A != max, A != median.

% @constraint Temporal only supports min and max.
hard(aggregate_t_valid,E,A) :- type(E,temporal), aggregate(E,A), A != min, A != max.

% @constraint Cannot aggregate nominal.
hard(aggregate_nominal,E) :- aggregate(E,_), type(E,nominal).

% @constraint Detail cannot be aggregated.
hard(aggregate_detail,E) :- channel(E,detail), aggregate(E,_).

% @constraint Shape requires discrete and not ordered (nominal). Using ordinal would't make a difference in Vega-Lite.
hard(shape_discrete_non_ordered,E) :- channel(E,shape), not type(E,nominal).

% @constraint Detail requires nominal.
hard(detail_non_ordered,E) :- channel(E,detail), not type(E,nominal).

% @constraint Size implies order so nominal is misleading.
hard(size_nominal) :- channel(E,size), type(E,nominal).

% @constraint Do not use size when data is negative as size implies that data is positive.
hard(size_negative,E) :- channel(E,size), enc_extent(E,MIN,MAX), MIN < 0, MAX > 0.

% === Across encodings and between encodings and marks ===

% @constraint Cannot use single channels twice. (added)
hard(repeat_channel,C):- single_channel(C), 2 { channel(_,C) }.

% @constraint There has to be at least one encoding. Otherwise, the visualization doesn't show anything.
hard(no_encodings) :- not encoding(_).

% @constraint Row and column require discrete. (added)
hard(row_or_column_c) :- channel_continuous(row;column).

% @constraint Don't use row without y. Just using y is simpler.
hard(row_no_y) :- channel(_,row), not channel(_,y).

% @constraint Don't use column without x. Just using x is simpler.
hard(column_no_x) :- channel(_,column), not channel(_,x).

% @constraint All encodings (if they have a channel) require field except if we have a count aggregate.
hard(encoding_no_field_and_not_count,E) :- not field(E,_), not aggregate(E,count), encoding(E).

% @constraint Count has to be quantitative and not use a field. (added)
% hard(count_q_without_field,E) :- aggregate(E,count), field(E,_).
% hard(count_q_without_field,E) :- aggregate(E,count), not type(E,quantitative).

% @constraint Count should not have a field. Having a field doesn't make a difference.
hard(count_with_field,E) :- aggregate(E,count), field(E,_).

% @constraint Text mark requires text channel.
hard(text_mark_without_text_channel) :- mark(text), not channel(_,text).

% @constraint Text channel requires text mark.
hard(text_channel_without_text_mark) :- channel(_,text), not mark(text).

% @constraint Point, tick, and bar require x or y channel.
hard(point_tick_bar_without_x_or_y) :- mark(point;tick;bar), not channel(_,x), not channel(_,y).

% @constraint Line and area require x and y channel.
hard(line_area_without_x_y) :- mark(line;area), not channel(_,(x;y)).

% @constraint pie chart must have a least one nominal field
hard(pie_without_nominal_field) :- mark(arc), not channel(_, nominal).

% @constraint Line and area cannot have two discrete. (added)
hard(line_area_with_discrete) :- mark(line;area), channel_discrete(x), channel_discrete(y).

% @constraint Bar and tick cannot have both x and y continuous.
hard(bar_tick_continuous_x_y) :- mark(bar;tick), channel_continuous(x), channel_continuous(y).

% @constraint Bar, tick, line, area require some continuous variable on x or y. (added)
hard(bar_tick_area_line_without_continuous_x_y) :- mark(bar;tick;area;line), not channel_continuous(x), not channel_continuous(y).

% @constraint Bar and area mark requires scale of continuous to start at zero.
% hard(bar_area_without_zero) :- mark(bar;area), channel(E,x), orientation(horizontal), not zero(E).
% hard(bar_area_without_zero) :- mark(bar;area), channel(E,y), orientation(vertical), not zero(E).

% @constraint Shape channel requires point mark.
hard(shape_without_point) :- channel(_,shape), not mark(point).

% @constraint Size only works with some marks. Vega-Lite can also size lines, and ticks but that would violate best practices.
hard(size_without_point_text) :- channel(_,size), not mark(point), not mark(text).

% @constraint Detail requires aggregation. Detail adds a field to the group by. Detail could also be used to add information to tooltips. We may remove this later.
hard(detail_without_agg) :- channel(_,detail), not aggregate(_,_).

% @constraint Do not use log for bar or area mark as they are often misleading. We may remove this rule in the future.
% hard(area_bar_with_log) :- mark(bar;area), log(E), channel(E,(x;y)).

% @constraint Rect mark needs discrete x and y.
hard(rect_without_d_d) :- mark(rect), not is_d_d.

% @constraint Don't use the same field on x and y.
hard(same_field_x_and_y) :- { field(E,F) : channel(E,x); field(E,F) : channel(E,y) } >= 2, field(F).

% @constraint If we use aggregation, then all continuous fields need to be aggeragted.
hard(aggregate_not_all_continuous):- aggregate(_,_), continuous(E), not aggregate(E,_).

% @constraint [Duplicate] Don't use count on x and y.
% hard(count_on_x_and_y):- channel(EX,x), channel(EY,y), aggregate(EX,count), aggregate(EY,count).

% @constraint Don't use count twice.
hard(count_twice) :- { aggregate(_,count) } >= 2.

% === Global properties ===

% @constraint Bars and area cannot overlap.
hard(bar_area_overlap) :- mark(bar;area), overlap.

% @constraint Rects shouldn't overlap. They are used for dioscrete heatmaps.
hard(rect_overlap) :- mark(rect), overlap.

% == Stacking ==

% @constraint Only use stacking for bar and area.
hard(stack_without_bar_area) :- stack(_), not mark(bar), not mark(area).

% @constraint Don't stack if aggregation is not summative (summative are count, sum, distinct, valid, missing).
hard(stack_without_summative_agg,E,A) :- stack(E,_), aggregate(E,A), not summative_aggregate_op(A).

% @constraint Need to stack if we use bar, area with discrete color.
hard(no_stack_with_bar_area_discrete_color,E) :- mark(bar;area), channel(E,color), discrete(E), not hide_axis(_), not stack(_).
%hard(no_stack_with_bar_area_discrete_color,E) :- mark(bar;area), channel(E,color), discrete(E), not stack(_).

% @constraint Can only use stack if we also use discrete color, or detail.
hard(stack_without_discrete_color_or_detail) :- stack(_), not channel_discrete(color), not channel(_,detail).

% @constraint If we use stack and detail, we also have to use quantitative color.
hard(stack_detail_without_q_color) :- stack(_), channel(_,detail), not channel(_,color).
hard(stack_detail_without_q_color,E) :- stack(_), channel(_,detail), channel(E,color), not aggregate(E,_).

% @constraint Stack can only be on continuous.
hard(stack_discrete,E) :- stack(E,_), discrete(E).

% @constraint Stack can only be on x or y.
hard(stack_without_x_y,E) :- stack(E,_), not channel(E,x), not channel(E,y).

% @constraint Cannot use non positional continuous with stack unless it's aggregated.
hard(stack_with_non_positional_non_agg,E,C) :- stack(_), non_positional(C), channel(E,C), not aggregate(E,_), continuous(E).

% @constraint Vega-Lite currently supports 8 shapes.
hard(shape_with_cardinality_gt_eight,E,C) :- channel(E,shape), enc_cardinality(E,C), C > 8.

% @constraint At most 20 categorical colors. (added)
hard(color_with_cardinality_gt_twenty,E,C) :- channel(E,color), discrete(E), enc_cardinality(E,C), C > 20.

% === Type checks ===

% @constraint Check mark.
hard(invalid_mark,M) :- mark(M), not marktype(M).

% @constraint Check types of encoding properties.
hard(invalid_channel,C) :- channel(_,C), not channel(C).
hard(invalid_field,F) :- field(_,F), not field(F).
hard(invalid_type,T) :- type(_,T), not type(T).
hard(invalid_agg,A) :- aggregate(_,A), not aggregate_op(A).
hard(invalid_bin,B) :- bin(_,B), not B >= 0.  % @constraint Bin has to be a natural number.

% @constraint Fieldtype has to be primitive type.
hard(invalid_fieldtype,T) :- fieldtype(_,T), not primitive_type(T).

% @constraint Task has to be one of the tasks.
hard(invalid_task,T) :- task(T), not tasks(T).

% @constraint Num_rows has to be larger than 0.
hard(invalid_num_rows,S) :- num_rows(S), S < 0.

% @constraint Cardinality has to be larger than 0.
hard(invalid_cardinality,C) :- cardinality(_,C), C < 0.

% @constraint Entropy has to be positive.
hard(invalid_entropy,E) :- entropy(_,E), E < 0.

% @constraint Extent only allowed for numbers (for now).
hard(invalid_extent_non_number,F) :- extent(F,_,_), not fieldtype(F,number).

% @constraint Order has to be correct.
hard(invalid_extent_order,MIN,MAX):- extent(_,MIN,MAX), MIN > MAX.

% @constraint The name of a field cannot be the name of an encoding. This is to prevent errors coming from the shortcuts in define.lp.
hard(encoding_field_same_name,N) :- encoding(N), field(N).

% === Other high-level constraints ===

% @constraint Do not use point in 1-D plots
hard(point_vertical_1d) :- mark(point), not channel(_, x).
hard(point_horizontal_1d) :- mark(point), not channel(_, y).

% @constraint Do not use color encoding for continuous field
hard(no_color_continuous) :- channel(E, color), continuous(E).

% @constraint norminal/ordinal/binned fields on x should not have cardinality < 8 (to break symmetry) (added)
% hard(nominal_x_low_cardinality_1) :- channel(EX,x), channel(EY,y), type(EX,nominal), type(EY,quantitative), enc_cardinality(EX,C), C < 6.
% hard(ordinal_x_low_cardinality_1) :- channel(EX,x), channel(EY,y), type(EX,ordinal), type(EY,quantitative), enc_cardinality(EX,C), C < 6.
% hard(binned_x_low_cardinality_1) :- channel(EX,x), channel(EY,y), bin(EX,_), type(EY,quantitative), enc_cardinality(EX,C), C < 6.
% hard(nominal_x_low_cardinality_2) :- channel(EX,x), channel(EY,y), type(EX,nominal), not type(EY,quantitative), enc_cardinality(EX,CX), enc_cardinality(EY,CY), CX < CY.
% hard(ordinal_x_low_cardinality_2) :- channel(EX,x), channel(EY,y), type(EX,ordinal), not type(EY,quantitative), enc_cardinality(EX,CX), enc_cardinality(EY,CY), CX < CY.
% hard(binned_x_low_cardinality_2) :- channel(EX,x), channel(EY,y), bin(EX,_), not type(EY,quantitative), enc_cardinality(EX,CX), enc_cardinality(EY,CY), CX < CY.
hard(x_low_cardinality_1) :- channel(EX,x), channel(EY,y), not type(EX,quantitative), type(EY,quantitative), enc_cardinality(EX,C), C < 3.
hard(x_low_cardinality_2) :- channel(EX,x), channel(EY,y), not type(EX,quantitative), not type(EY,quantitative), enc_cardinality(EX,CX), enc_cardinality(EY,CY), CX < CY.
hard(binned_x_low_cardinality_1) :- channel(EX,x), channel(EY,y), bin(EX,_), type(EY,quantitative), enc_cardinality(EX,C), C < 6.
hard(binned_x_low_cardinality_2) :- channel(EX,x), channel(EY,y), bin(EX,_), not type(EY,quantitative), enc_cardinality(EX,CX), enc_cardinality(EY,CY), CX < CY.

% @constraint
% size_unaggregated :- channel(E,size), { channel(_,_) } != 3, not aggregate(E,_).

% hide-axis related
hard(allow_hide_axis_on_x) :- hide_axis(E), not channel(E,x).
hard(allow_hide_axis_on_nominal) :- hide_axis(E), not type(E,nominal).
hard(allow_hide_axis_when_color_present) :- hide_axis(_), not channel(_, color).
hard(allow_hide_axis_when_column_present) :- hide_axis(_), not channel(_, column).
hard(allow_hide_axis_on_field_same_as_color) :- hide_axis(E1), field(E1,F1), field(E2,F2), channel(E2,color), F1 != F2.

% channel cannot present without encoding
hard(channel_encoding_both_present) :- channel(E,_), not encoding(E).
hard(aggregate_encoding_both_present) :- aggregate(E,_), not encoding(E).
hard(type_encoding_both_present) :- type(E,_), not encoding(E).