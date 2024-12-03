module RuleCheck_top (
    input wire clk,
    input wire rst,
    input wire interboard_rst,
    input wire [8*18*6-1:0] map,

    output wire rule_valid
);

    wire [6*18*6-1:0] table_map = map[6*18*6-1:0]; // 檯面

endmodule