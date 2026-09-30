// Package sysmem reports physical memory, to warn when the weights a plan
// keeps in RAM won't fit and would be paged from disk.
package sysmem

// Info is physical memory in bytes. OK is false when it couldn't be read.
type Info struct {
	Total, Available uint64
	OK               bool
}
