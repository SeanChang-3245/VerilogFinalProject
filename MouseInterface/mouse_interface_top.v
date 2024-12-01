module MouseInterface_top (
    input wire clk,
    input wire rst,
    
    inout wire PS2_CLK,          // PS2 Mouse
    inout wire PS2_DATA,         // PS2 Mouse
    
    output wire mouse_valid,     // 處理滑鼠不在任何一個區塊的情況
    output wire l_click,
    output wire cheat_activate,
    output wire [9:0] mouse_x,
    output wire [8:0] mouse_y,
    output wire [4:0] mouse_block_x,
    output wire [2:0] mouse_block_y
);



endmodule