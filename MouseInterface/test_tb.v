module test_tb (
	input wire clk,
    input wire rst,
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA, 
	output wire [6:0] DISPLAY,
	output wire [3:0] DIGIT
);	

    wire [11:0] mx, my;
    wire [2:0] btn_click; 

	ps2_mouse_xy inst(
        .clk(clk),
        .reset(rst),
        .ps2_clk(PS2_CLK),
        .ps2_data(PS2_DATA),
        .mx(mx),
        .my(my),
        .btn_click(btn_click)
    );

    wire [3:0] x_hundred, x_ten, x_one;
    assign x_hundred = mx / 100;
    assign x_ten = (mx / 10) % 10;
    assign x_one = mx % 10;

    SevenSegment ss (
        .clk(clk),
        .rst(rst),
        .nums({4'b0001, x_hundred, x_ten, x_one}),
        .display(DISPLAY),
        .digit(DIGIT)
    );

endmodule