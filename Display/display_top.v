module Display_top(
    input wire clk,
    input wire rst,
    input wire interboard_rst,
    input wire [9:0] mouse_x,
    input wire [8:0] mouse_y,
    input wire [8*18-1:0] sel_card,
    input wire [8*18*6-1:0] map,

    output wire hsync,
    output wire vsync,
    output wire [3:0] vgaRed,
    output wire [3:0] vgaGreen,
    output wire [3:0] vgaBlue
);

    wire clk_25MHz;
    clock_divider m2 (.clk(clk), .clk_div(clk_25MHz));

    wire [9:0] h_cnt, v_cnt;

    vga_controller vga_inst(
        .pclk(clk_25MHz),
        .reset(rst),
        .hsync(hsync),
        .vsync(vsync),
        .valid(valid),
        .h_cnt(h_cnt),
        .v_cnt(v_cnt)
    );


    // use if-else to determine which object should be drawn on the top
    // each object should output a signal to indicate whether is should be drawn at this pixel
    // priority: mouse > card = button > background
    // e.g.
    // if(mouse_valid) begin
    //     pixel = mouse_pixel
    // end
    // else if(card_valid) begin
    //     pixel = card_pixel
    // end
    // else if(button_valid) begin
    //     pixel = button_pixel
    // end
    // else begin
    //     pixel = bg_pixel
    // end




endmodule