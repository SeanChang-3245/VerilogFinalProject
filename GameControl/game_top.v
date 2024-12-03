module GameControl_top #(
    parameter PLAYER = 0
)(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    input wire start_game,
    input wire rule_valid,
    input wire mouse_valid,
    input wire cheat_activate,
    input wire move_left,
    input wire move_right,
    input wire reset_table,
    input wire done_and_next,
    input wire draw_and_next,
    input wire [105:0] available_card,   // used to determine which cards can be drawn
    input wire [5:0] picked_card,        // from memory, indicate the card that is clicked
    input wire [9:0] mouse_x,
    input wire [8:0] mouse_y,
    input wire [4:0] mouse_block_x,      // mouse information
    input wire [2:0] mouse_block_y,      // mouse information

    output wire ctrl_en,
    output wire ctrl_move_dir,
    output wire [4:0] ctrl_block_x,      // protocol information 
    output wire [2:0] ctrl_block_y,      // protocol information
    output wire [3:0] ctrl_msg_type,
    output wire [5:0] ctrl_card,
    output wire [2:0] ctrl_sel_len,

    output wire [8*18-1:0] sel_card      // to Display, indicate which cards are selected
);


    // Message type definition
    localparam TABLE_TAKE = 0;
    localparam TABLE_DOWN = 1;
    localparam TABLE_SHIFT = 2;

    localparam HAND_TAKE = 3;
    localparam HAND_DRAW = 4;

    localparam DECK_DRAW = 5;

    localparam STATE_TURN = 6;
    localparam STATE_RST_TABLE = 7;
    localparam STATE_RST_GAME = 8;
    localparam STATE_CHEAT = 9;



endmodule