module P1_top (
    input wire clk,
    input wire rst,
    input wire btnL,
    input wire btnR,
    input wire [15:0] SW,
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse
    output wire [3:0] DIGIT,     // 7-segment display
    output wire [6:0] DISPLAY,   // 7-segment display
    output wire [3:0] vgaRed,    // VGA
    output wire [3:0] vgaGreen,  // VGA
    output wire [3:0] vgaBlue,   // VGA
    output wire hsync,           // VGA
    output wire vsync            // VGA
);
    // Game Control 要把這個用 parameter 傳進去 
    localparam PLAYER = 0;

    // 所有 rst 都要換成 rstGame
    // rstGame 包含這張板子的 rst 和從interboard來的 rst
    wire rstGame = rst;


endmodule