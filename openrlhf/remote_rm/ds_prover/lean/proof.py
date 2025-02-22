import re
import json

import numpy as np

from openrlhf.remote_rm.ds_prover.utils import AttrDict, LEAN4_DEFAULT_HEADER


class Proof(object):
    def __init__(self, full_code, _args, _result_backup=None, **kwargs):
        self._kwargs_backup = kwargs
        for key, val in kwargs.items():
            self.__setattr__(key, val)
        self._args = _args
        self._update_full_code(full_code, _result_backup=_result_backup)
    
    @property
    def result(self):
        if self._verifier_request_id is not None:
            self._result = self._scheduler.get_request_outputs(self._verifier_request_id)
            self._verifier_request_id = None
        return self._result
    
    def is_result_ready(self):
        if self._verifier_request_id is None:
            return True
        status = self._scheduler.get_request_status(self._verifier_request_id)
        if status is not None:
            self._result = status
            self._verifier_request_id = None
        return self._result is not None
    
    @property
    def cleaned_code(self):
        return self.full_code[len(self.header) + len(self.formal_statement): len(self.full_code) - len(self.tailer)]
    
    def _update_full_code(self, full_code, _result_backup=None):
        self.full_code = full_code
        self._verifier_request_id, self._result = None, None
        if _result_backup is not None:
            self._result = _result_backup
        elif self._args.require_verification:  # need to call verification server
            self._verifier_request_id = self._scheduler.submit_request(dict(
                code=self.full_code,
                ast=True, tactics=True,
            ))
        self._parse_full_code_lines()
    
    def _parse_full_code_lines(self):
        self._full_code_lines = self.full_code.split('\n')
        self._line_offset, _offset = [], -1
        for _line in self._full_code_lines:
            _offset += 1  # '\n'
            self._line_offset.append(_offset)
            _offset += len(_line)
    
    def _get_idx(self, pos_info):
        return self._line_offset[pos_info['line'] - 1] + pos_info['column']
    
    def segmentation(self, result=None):
        if result is None:
            result = self.result
        if 'errors' not in result:
            # compiler timeout
            return []
        
        _prefix_len = len(self.header) + len(self.formal_statement)
        truncate_pos = len(self.full_code) - len(self.tailer)
        for info in result['sorries'] + result['errors']:
            info_pos = self._get_idx(info['pos'])
            if info_pos >= _prefix_len and not info.get('data', str()).lstrip().startswith('unsolved goals'):
                truncate_pos = min(truncate_pos, info_pos)
        partial_code = self.full_code[:truncate_pos]
        if len(partial_code) <= _prefix_len:
            # all proof lines are invalid
            return []
        
        code_lines = partial_code.split('\n')
        pos_last, segments = _prefix_len, []
        for line_idx in range(len(code_lines)):
            if self._line_offset[line_idx] >= _prefix_len:
                def compute_last_valid_char_pos(line):
                    idx, last_non_blank = 0, len(line) + 1
                    while idx < len(line):
                        if line[idx: idx+2] == '--':
                            return last_non_blank
                        elif line[idx: idx+2] == '/-':
                            if '-/' not in line[idx+2:]:
                                # cannot split in this line
                                return len(line) + 1
                            idx = line.find('-/', idx+2) + 1
                        elif line[idx] != ' ':
                            last_non_blank = idx
                        idx += 1
                    return last_non_blank
                line_lastChar = self._line_offset[line_idx] + compute_last_valid_char_pos(code_lines[line_idx])
                line_endPos = self._line_offset[line_idx] + len(code_lines[line_idx])

                pos_min, goal = 1e9, None
                for tactic_info in result['ast']['tactics']:
                    pos, endPos = tactic_info['pos'], tactic_info['endPos']
                    if line_lastChar <= endPos and endPos <= line_endPos and pos < pos_min:
                        pos_min = pos
                        goal = tactic_info['stateAfter']
                if goal is not None:
                    for tactic_info in result['ast']['tactics']:
                        pos, endPos = tactic_info['pos'], tactic_info['endPos']
                        if pos_last < endPos and endPos <= line_endPos and pos < pos_min:
                            pos_min = pos

                    while pos_min > 0 and partial_code[pos_min - 1] != '\n':
                        pos_min -= 1
                    indent_len = 0
                    while partial_code[pos_min + indent_len] == ' ':
                        indent_len += 1
                    newline_with_indent = '\n' + ' ' * indent_len
                    
                    segments.append(AttrDict(
                        tactic_code=partial_code[pos_last: line_endPos] + '\n',
                        state_comment=newline_with_indent.join([
                            ' ' * indent_len + '/- tactic state:',
                            '  ' + goal.replace('\n', newline_with_indent + '  '),
                            '-/\n'
                        ]),
                        goal=goal,
                        indent=indent_len,
                    ))
                    pos_last = line_endPos + 1
        if result['complete'] and (len(segments) == 0 or segments[-1].goal != 'no goals' or segments[-1].indent != segments[0].indent):
            indent_len = 2 if len(segments) == 0 else segments[0].indent
            newline_with_indent = '\n' + ' ' * indent_len
            segments.append(AttrDict(
                tactic_code=partial_code[pos_last:].rstrip(' \n') + '\n',
                state_comment=newline_with_indent.join([
                    ' ' * indent_len + '/- tactic state:',
                    '  no goals',
                    '-/\n'
                ]),
                goal='no goals',
                indent=indent_len,
            ))
        segments = [seg for seg in segments if len(seg.tactic_code.strip(' \n')) > 0]
        return segments


class ProofSummarizer(object):
    def __init__(self, data, scheduler=None):
        """
        Inputs:
            data (`dict`): The problem information storing in a `dict` object.
                formal_statement (`str`): The formal statement of the unproved problem;
                header (`str`, *optional*, defaults to ''): The code header required by the complier;
                tailer (`str`, *optional*, defaults to ''): The code tailer required by the complier.
            scheduler (prover.workers.scheduler.Scheduler, *optional*, defaults to None):
                An interface to submit requests to models and the verification server.
                If set to None, the downstream tasks may require the verification result as inputs.
        """
        self.formal_statement = data['formal_statement']
        self.header = data.get('header', LEAN4_DEFAULT_HEADER)
        self.tailer = data.get('tailer', str())
        self.scheduler = scheduler
    
    def analyze(self, code, require_verification=True):
        """
        Inputs:
            code (`str`): The code of formal proof.
            require_verification (`bool`, *optional*, defaults to True):
                Whether to submit a request to the verification server.
                If set to False, the downstream tasks may require the verification result as inputs.
        Return:
            A `Proof` object that summarizes the code.
        """
        return Proof(
            full_code=''.join([self.header, self.formal_statement, code.rstrip(' \n'), self.tailer]),
            raw_code=code,
            formal_statement=self.formal_statement,
            header=self.header,
            tailer=self.tailer,
            _scheduler=self.scheduler,
            _args=AttrDict(
                require_verification=require_verification,
            )
        )
