import tkinter as tk

root = tk.Tk()
root.title("Panel")

# LERPanel
ler_panel_frame = tk.Frame(root)
ler_panel_frame.pack()

ler_label = tk.Label(ler_panel_frame, text="Line Edge Roughness")
ler_label.grid(row=0, column=0, sticky="w", columnspan=2)

no_of_measured_edges_label = tk.Label(ler_panel_frame, text="No. of measured edges:")
no_of_measured_edges_label.grid(row=1, column=0, sticky="w")
no_of_measured_edges_value = len(resamplededgelinesdata)
no_of_measured_edges_value_label = tk.Label(ler_panel_frame, text=str(no_of_measured_edges_value))
no_of_measured_edges_value_label.grid(row=1, column=1, sticky="w")

cutoff_wavelengths_label = tk.Label(ler_panel_frame, text="Cutoff wavelengths:")
cutoff_wavelengths_label.grid(row=2, column=0, sticky="w")
lambda_min = round(2 * scale, 0.01)
lambda_max = round(scale * (xmax - xmin + 1), 0.01)
cutoff_wavelengths_value_label = tk.Label(ler_panel_frame, text=f"λ_min = {lambda_min}, λ_max = {lambda_max} nm")
cutoff_wavelengths_value_label.grid(row=2, column=1, sticky="w")

median_ler_label = tk.Label(ler_panel_frame, text="Median LER:")
median_ler_label.grid(row=3, column=0, sticky="w")
median_ler_value = round(3 * scale * math.sqrt(statistics.median(edgeVariances)), 0.01)
median_ler_value_label = tk.Label(ler_panel_frame, text=f"3σ_ε = {median_ler_value} nm")
median_ler_value_label.grid(row=3, column=1, sticky="w")

ler_range_label = tk.Label(ler_panel_frame, text="LER 3σ_e range:")
ler_range_label.grid(row=4, column=0, sticky="w")
min_ler_value = round(3 * scale * math.sqrt(min(edgeVariances)), 0.01)
max_ler_value = round(3 * scale * math.sqrt(max(edgeVariances)), 0.01)
ler_range_value_label = tk.Label(ler_panel_frame, text=f"{min_ler_value} - {max_ler_value} nm")
ler_range_value_label.grid(row=4, column=1, sticky="w")

# whiteLWRPanel
white_lwr_panel_frame = tk.Frame(root)
white_lwr_panel_frame.pack()

white_lwr_label = tk.Label(white_lwr_panel_frame, text="White Line Width Roughness")
white_lwr_label.grid(row=0, column=0, sticky="w", columnspan=2)

no_of_measured_lines_label = tk.Label(white_lwr_panel_frame, text="No. of measured lines:")
no_of_measured_lines_label.grid(row=1, column=0, sticky="w")
no_of_measured_lines_value = len(whiteWidthVariances)
no_of_measured_lines_value_label = tk.Label(white_lwr_panel_frame, text=str(no_of_measured_lines_value))
no_of_measured_lines_value_label.grid(row=1, column=1, sticky="w")

average_line_width_label = tk.Label(white_lwr_panel_frame, text="Average Line Width:")
average_line_width_label.grid(row=2, column=0, sticky="w")
average_line_width_value = round(scale * whiteLW, 0.01)
average_line_width_value_label = tk.Label(white_lwr_panel_frame, text=f"{average_line_width_value} nm")
average_line_width_value_label.grid(row=2, column=1, sticky="w")

line_duty_cycle_label = tk.Label(white_lwr_panel_frame, text="Line duty cycle:")
line_duty_cycle_label.grid(row=3, column=0, sticky="w")
line_duty_cycle_value = round(1 - dutycycle, 0.01)
line_duty_cycle_value_label = tk.Label(white_lwr_panel_frame, text=str(line_duty_cycle_value))
line_duty_cycle_value_label.grid(row=3, column=1, sticky="w")

median_lwr_label = tk.Label(white_lwr_panel_frame, text="Median LWR:")
median_lwr_label.grid(row=4, column=0, sticky="w")
median_lwr_value = round(3 * scale * math.sqrt(statistics.median(whiteWidthVariances)), 0.01)
median_lwr_value_label = tk.Label(white_lwr_panel_frame, text=f"3σ_w = {median_lwr_value} nm")
median_lwr_value_label.grid(row=4, column=1, sticky="w")

lwr_range_label = tk.Label(white_lwr_panel_frame, text="LWR 3σ_w range:")
lwr_range_label.grid(row=5, column=0, sticky="w")
min_lwr_value = round(3 * scale * math.sqrt(min(whiteWidthVariances)), 0.01)
max_lwr_value = round(3 * scale * math.sqrt(max(whiteWidthVariances)), 0.01)
lwr_range_value_label = tk.Label(white_lwr_panel_frame, text=f"{min_lwr_value} - {max_lwr_value} nm")
lwr_range_value_label.grid(row=5, column=1, sticky="w")

median_lin_corr_coeff_label = tk.Label(white_lwr_panel_frame, text="Median Lin. corr. coeff.:")
median_lin_corr_coeff_label.grid(row=6, column=0, sticky="w")
median_lin_corr_coeff_value = round(statistics.median(whitec), 0.01)
median_lin_corr_coeff_value_label = tk.Label(white_lwr_panel_frame, text=f"c_white = {median_lin_corr_coeff_value}")
median_lin_corr_coeff_value_label.grid(row=6, column=1, sticky="w")

lin_corr_coeff_range_label = tk.Label(white_lwr_panel_frame, text="c_white range:")
lin_corr_coeff_range_label.grid(row=7, column=0, sticky="w")
min_lin_corr_coeff_value = round(min(whitec), 0.01)
max_lin_corr_coeff_value = round(max(whitec), 0.01)
lin_corr_coeff_range_value_label = tk.Label(white_lwr_panel_frame, text=f"{min_lin_corr_coeff_value} - {max_lin_corr_coeff_value}")
lin_corr_coeff_range_value_label.grid(row=7, column=1, sticky="w")

# blackLWRPanel
black_lwr_panel_frame = tk.Frame(root)
black_lwr_panel_frame.pack()

black_lwr_label = tk.Label(black_lwr_panel_frame, text="Black Line Width Roughness")
black_lwr_label.grid(row=0, column=0, sticky="w", columnspan=2)

no_of_measured_lines_label = tk.Label(black_lwr_panel_frame, text="No. of measured lines:")
no_of_measured_lines_label.grid(row=1, column=0, sticky="w")
no_of_measured_lines_value = len(blackWidthVariances)
no_of_measured_lines_value_label = tk.Label(black_lwr_panel_frame, text=str(no_of_measured_lines_value))
no_of_measured_lines_value_label.grid(row=1, column=1, sticky="w")

average_line_width_label = tk.Label(black_lwr_panel_frame, text="Average Line Width:")
average_line_width_label.grid(row=2, column=0, sticky="w")
average_line_width_value = round(scale * blackLW, 0.01)
average_line_width_value_label = tk.Label(black_lwr_panel_frame, text=f"{average_line_width_value} nm")
average_line_width_value_label.grid(row=2, column=1, sticky="w")

line_duty_cycle_label = tk.Label(black_lwr_panel_frame, text="Line duty cycle:")
line_duty_cycle_label.grid(row=3, column=0, sticky="w")
line_duty_cycle_value = round(1 - dutycycle, 0.01)
line_duty_cycle_value_label = tk.Label(black_lwr_panel_frame, text=str(line_duty_cycle_value))
line_duty_cycle_value_label.grid(row=3, column=1, sticky="w")

median_lwr_label = tk.Label(black_lwr_panel_frame, text="Median LWR:")
median_lwr_label.grid(row=4, column=0, sticky="w")
median_lwr_value = round(3 * scale * math.sqrt(statistics.median(blackWidthVariances)), 0.01)
median_lwr_value_label = tk.Label(black_lwr_panel_frame, text=f"3σ_w = {median_lwr_value} nm")
median_lwr_value_label.grid(row=4, column=1, sticky="w")

lwr_range_label = tk.Label(black_lwr_panel_frame, text="LWR 3σ_w range:")
lwr_range_label.grid(row=5, column=0, sticky="w")
min_lwr_value = round(3 * scale * math.sqrt(min(blackWidthVariances)), 0.01)
max_lwr_value = round(3 * scale * math.sqrt(max(blackWidthVariances)), 0.01)
lwr_range_value_label = tk.Label(black_lwr_panel_frame, text=f"{min_lwr_value} - {max_lwr_value} nm")
lwr_range_value_label.grid(row=5, column=1, sticky="w")

median_lin_corr_coeff_label = tk.Label(black_lwr_panel_frame, text="Median Lin. corr. coeff.:")
median_lin_corr_coeff_label.grid(row=6, column=0, sticky="w")
median_lin_corr_coeff_value = round(statistics.median(blackc), 0.01)
median_lin_corr_coeff_value_label = tk.Label(black_lwr_panel_frame, text=f"c_black = {median_lin_corr_coeff_value}")
median_lin_corr_coeff_value_label.grid(row=6, column=1, sticky="w")

lin_corr_coeff_range_label = tk.Label(black_lwr_panel_frame, text="c_black range:")
lin_corr_coeff_range_label.grid(row=7, column=0, sticky="w")
min_lin_corr_coeff_value = round(min(blackc), 0.01)
max_lin_corr_coeff_value = round(max(blackc), 0.01)
lin_corr_coeff_range_value_label = tk.Label(black_lwr_panel_frame, text=f"{min_lin_corr_coeff_value} - {max_lin_corr_coeff_value}")
lin_corr_coeff_range_value_label.grid(row=7, column=1, sticky="w")

# whiteLPRpanel
white_lpr_panel_frame = tk.Frame(root)
white_lpr_panel_frame.pack()

white_lpr_label = tk.Label(white_lpr_panel_frame, text="White Line Placement Accuracy")
white_lpr_label.grid(row=0, column=0, sticky="w", columnspan=2)

no_of_measured_lines_label = tk.Label(white_lpr_panel_frame, text="No. of measured lines:")
no_of_measured_lines_label.grid(row=1, column=0, sticky="w")
no_of_measured_lines_value = len(whitePlacementVariances)
no_of_measured_lines_value_label = tk.Label(white_lpr_panel_frame, text=str(no_of_measured_lines_value))
no_of_measured_lines_value_label.grid(row=1, column=1, sticky="w")

placement_roughness_label = tk.Label(white_lpr_panel_frame, text="Placement Roughness:")
placement_roughness_label.grid(row=2, column=0, sticky="w")
placement_roughness_value = round(3 * scale * math.sqrt(statistics.median(whitePlacementVariances)), 0.01)
placement_roughness_value_label = tk.Label(white_lpr_panel_frame, text=f"3σ_p = {placement_roughness_value} nm")
placement_roughness_value_label.grid(row=2, column=1, sticky="w")

placement_roughness_range_label = tk.Label(white_lpr_panel_frame, text="Placement 3σ_p range:")
placement_roughness_range_label.grid(row=3, column=0, sticky="w")
min_placement_roughness_value = round(3 * scale * math.sqrt(min(whitePlacementVariances)), 0.01)
max_placement_roughness_value = round(3 * scale * math.sqrt(max(whitePlacementVariances)), 0.01)
placement_roughness_range_value_label = tk.Label(white_lpr_panel_frame, text=f"{min_placement_roughness_value} - {max_placement_roughness_value} nm")
placement_roughness_range_value_label.grid(row=3, column=1, sticky="w")

cross_line_xi_label = tk.Label(white_lpr_panel_frame, text="Cross-line ξ_┴")
cross_line_xi_label.grid(row=4, column=0, sticky="w")
cross_line_xi_value = round(ξ / whitePerpCorr, 0.01)
cross_line_xi_value_label = tk.Label(white_lpr_panel_frame, text=f"ξ_┴ = {cross_line_xi_value}, L_o; A_o = {A / whitePerpCorr}")
cross_line_xi_value_label.grid(row=4, column=1, sticky="w")

in_line_xi_label = tk.Label(white_lpr_panel_frame, text="In-line ξ_||")
in_line_xi_label.grid(row=5, column=0, sticky="w")
in_line_xi_value = round(scale * ξ / whiteParCorr, 0.01)
in_line_xi_value_label = tk.Label(white_lpr_panel_frame, text=f"ξ_|| = {in_line_xi_value} nm")
in_line_xi_value_label.grid(row=5, column=1, sticky="w")

pitch_label = tk.Label(white_lpr_panel_frame, text="Pitch")
pitch_label.grid(row=6, column=0, sticky="w")
pitch_value = round(m * scale, 0.01) / centroidPitchandWalk
pitch_value_label = tk.Label(white_lpr_panel_frame, text=f"Lo = {pitch_value} nm")
pitch_value_label.grid(row=6, column=1, sticky="w")

pitch_walking_label = tk.Label(white_lpr_panel_frame, text="Pitch walking")
pitch_walking_label.grid(row=7, column=0, sticky="w")
pitch_walking_value = pw / centroidPitchandWalk
pitch_walking_value_label = tk.Label(white_lpr_panel_frame, text=f"pw = {pitch_walking_value} Lo")
pitch_walking_value_label.grid(row=7, column=1, sticky="w")

# blackLPRPanel
black_lpr_panel_frame = tk.Frame(root)
black_lpr_panel_frame.pack()

black_lpr_label = tk.Label(black_lpr_panel_frame, text="Black Line Placement Accuracy")
black_lpr_label.grid(row=0, column=0, sticky="w", columnspan=2)

no_of_measured_lines_label = tk.Label(black_lpr_panel_frame, text="No. of measured lines:")
no_of_measured_lines_label.grid(row=1, column=0, sticky="w")
no_of_measured_lines_value = len(blackPlacementVariances)
no_of_measured_lines_value_label = tk.Label(black_lpr_panel_frame, text=str(no_of_measured_lines_value))
no_of_measured_lines_value_label.grid(row=1, column=1, sticky="w")

placement_roughness_label = tk.Label(black_lpr_panel_frame, text="Placement Roughness:")
placement_roughness_label.grid(row=2, column=0, sticky="w")
placement_roughness_value = round(3 * scale * math.sqrt(statistics.median(blackPlacementVariances)), 0.01)
placement_roughness_value_label = tk.Label(black_lpr_panel_frame, text=f"3σ_p = {placement_roughness_value} nm")
placement_roughness_value_label.grid(row=2, column=1, sticky="w")

placement_roughness_range_label = tk.Label(black_lpr_panel_frame, text="Placement 3σ_p range:")
placement_roughness_range_label.grid(row=3, column=0, sticky="w")
min_placement_roughness_value = round(3 * scale * math.sqrt(min(blackPlacementVariances)), 0.01)
max_placement_roughness_value = round(3 * scale * math.sqrt(max(blackPlacementVariances)), 0.01)
placement_roughness_range_value_label = tk.Label(black_lpr_panel_frame, text=f"{min_placement_roughness_value} - {max_placement_roughness_value} nm")
placement_roughness_range_value_label.grid(row=3, column=1, sticky="w")

cross_line_xi_label = tk.Label(black_lpr_panel_frame, text="Cross-line ξ_┴")
cross_line_xi_label.grid(row=4, column=0, sticky="w")
cross_line_xi_value = round(ξ / blackPerpCorr, 0.01)
cross_line_xi_value_label = tk.Label(black_lpr_panel_frame, text=f"ξ_┴ = {cross_line_xi_value}, L_o; A_o = {A / blackPerpCorr}")
cross_line_xi_value_label.grid(row=4, column=1, sticky="w")

in_line_xi_label = tk.Label(black_lpr_panel_frame, text="In-line ξ_||")
in_line_xi_label.grid(row=5, column=0, sticky="w")
in_line_xi_value = round(scale * ξ / blackParCorr, 0.01)
in_line_xi_value_label = tk.Label(black_lpr_panel_frame, text=f"ξ_|| = {in_line_xi_value} nm")
in_line_xi_value_label.grid(row=5, column=1, sticky="w")

resultstable_frame = tk.Frame(root)
resultstable_frame.pack()

resultstable_label = tk.Label(resultstable_frame, text="Result Table")
resultstable_label.pack()

ler_panel_frame.pack()
white_lwr_panel_frame.pack()
black_lwr_panel_frame.pack()
white_lpr_panel_frame.pack()
black_lpr_panel_frame.pack()
resultstable_frame.pack()

root.mainloop()