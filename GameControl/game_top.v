module GameControl_top #(
    parameter PLAYER = 0
)(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    input wire shift_en,                  // from Player_top, decide which operation the player is doing, shift or take/down
    input wire send_ready,                // from InterboardCommunication, indicate the module is ready to transmit action done to other side
    input wire start_game,
    input wire rule_valid,
    input wire mouse_inblock,
    input wire cheat_activate,
    input wire move_left,
    input wire move_right,
    input wire reset_table,
    input wire done_and_next,
    input wire draw_and_next,
    input wire interboard_en,
    input wire [3:0] interboard_msg_type, // from interboard, used to advance state in FSM
    input wire [105:0] available_card,    // used to determine which cards can be drawn
    input wire [8*18*6-1:0] map,
    // input wire [5:0] picked_card,         // from memory, indicate the card that is clicked
    input wire [9:0] mouse_x,
    input wire [8:0] mouse_y,
    input wire [4:0] mouse_block_x,       // mouse information
    input wire [2:0] mouse_block_y,       // mouse information

    output wire can_done,
    output wire can_draw,
    output wire transmit,                 // to InterboardCommunication and Memory, indicate it's this player's turn to transmit data (move)
    output wire ctrl_en,
    output wire ctrl_move_dir,
    output wire [4:0] ctrl_block_x,       // protocol information 
    output wire [2:0] ctrl_block_y,       // protocol information
    output wire [3:0] ctrl_msg_type,
    output wire [5:0] ctrl_card,
    output wire [2:0] ctrl_sel_len,

    output wire [8*18-1:0] sel_card       // to Display, indicate which cards are selected
);


    // Message type definition
    localparam TABLE_TAKE = 0;
    localparam TABLE_DOWN = 1;
    localparam TABLE_SHIFT = 2;

    localparam HAND_TAKE = 3;
    localparam HAND_DOWN = 4;
    // localparam HAND_DRAW = 4;

    localparam DECK_DRAW = 5;
    // localparam DECK_DOWN = 6;

    localparam STATE_TURN = 6;
    localparam STATE_RST_TABLE = 7;
    localparam STATE_CHEAT = 8;
    // localparam STATE_RST_GAME = 9;


    // use state to check whether this is my turn
    wire my_turn;

    // init draw use two states, player one draw 14 cards first, then pass a signal 
    // to player two to notify it is his turn to draw


endmodule